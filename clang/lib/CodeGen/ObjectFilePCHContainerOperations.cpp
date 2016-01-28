//===--- ObjectFilePCHContainerOperations.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetRegistry.h"
#include <memory>

using namespace clang;

#define DEBUG_TYPE "pchcontainer"

namespace {
class PCHContainerGenerator : public ASTConsumer {
  DiagnosticsEngine &Diags;
  const std::string MainFileName;
  ASTContext *Ctx;
  ModuleMap &MMap;
  const HeaderSearchOptions &HeaderSearchOpts;
  const PreprocessorOptions &PreprocessorOpts;
  CodeGenOptions CodeGenOpts;
  const TargetOptions TargetOpts;
  const LangOptions LangOpts;
  std::unique_ptr<llvm::LLVMContext> VMContext;
  std::unique_ptr<llvm::Module> M;
  std::unique_ptr<CodeGen::CodeGenModule> Builder;
  raw_pwrite_stream *OS;
  std::shared_ptr<PCHBuffer> Buffer;

  /// Visit every type and emit debug info for it.
  struct DebugTypeVisitor : public RecursiveASTVisitor<DebugTypeVisitor> {
    clang::CodeGen::CGDebugInfo &DI;
    ASTContext &Ctx;
    DebugTypeVisitor(clang::CodeGen::CGDebugInfo &DI, ASTContext &Ctx)
        : DI(DI), Ctx(Ctx) {}

    /// Determine whether this type can be represented in DWARF.
    static bool CanRepresent(const Type *Ty) {
      return !Ty->isDependentType() && !Ty->isUndeducedType();
    }

    bool VisitImportDecl(ImportDecl *D) {
      auto *Import = cast<ImportDecl>(D);
      if (!Import->getImportedOwningModule())
        DI.EmitImportDecl(*Import);
      return true;
    }

    bool VisitTypeDecl(TypeDecl *D) {
      // TagDecls may be deferred until after all decls have been merged and we
      // know the complete type. Pure forward declarations will be skipped, but
      // they don't need to be emitted into the module anyway.
      if (auto *TD = dyn_cast<TagDecl>(D))
        if (!TD->isCompleteDefinition())
          return true;

      QualType QualTy = Ctx.getTypeDeclType(D);
      if (!QualTy.isNull() && CanRepresent(QualTy.getTypePtr()))
        DI.getOrCreateStandaloneType(QualTy, D->getLocation());
      return true;
    }

    bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
      QualType QualTy(D->getTypeForDecl(), 0);
      if (!QualTy.isNull() && CanRepresent(QualTy.getTypePtr()))
        DI.getOrCreateStandaloneType(QualTy, D->getLocation());
      return true;
    }

    bool VisitFunctionDecl(FunctionDecl *D) {
      if (isa<CXXMethodDecl>(D))
        // This is not yet supported. Constructing the `this' argument
        // mandates a CodeGenFunction.
        return true;

      SmallVector<QualType, 16> ArgTypes;
      for (auto i : D->params())
        ArgTypes.push_back(i->getType());
      QualType RetTy = D->getReturnType();
      QualType FnTy = Ctx.getFunctionType(RetTy, ArgTypes,
                                          FunctionProtoType::ExtProtoInfo());
      if (CanRepresent(FnTy.getTypePtr()))
        DI.EmitFunctionDecl(D, D->getLocation(), FnTy);
      return true;
    }

    bool VisitObjCMethodDecl(ObjCMethodDecl *D) {
      if (!D->getClassInterface())
        return true;

      bool selfIsPseudoStrong, selfIsConsumed;
      SmallVector<QualType, 16> ArgTypes;
      ArgTypes.push_back(D->getSelfType(Ctx, D->getClassInterface(),
                                        selfIsPseudoStrong, selfIsConsumed));
      ArgTypes.push_back(Ctx.getObjCSelType());
      for (auto i : D->params())
        ArgTypes.push_back(i->getType());
      QualType RetTy = D->getReturnType();
      QualType FnTy = Ctx.getFunctionType(RetTy, ArgTypes,
                                          FunctionProtoType::ExtProtoInfo());
      if (CanRepresent(FnTy.getTypePtr()))
        DI.EmitFunctionDecl(D, D->getLocation(), FnTy);
      return true;
    }
  };

public:
  PCHContainerGenerator(CompilerInstance &CI, const std::string &MainFileName,
                        const std::string &OutputFileName,
                        raw_pwrite_stream *OS,
                        std::shared_ptr<PCHBuffer> Buffer)
      : Diags(CI.getDiagnostics()), Ctx(nullptr),
        MMap(CI.getPreprocessor().getHeaderSearchInfo().getModuleMap()),
        HeaderSearchOpts(CI.getHeaderSearchOpts()),
        PreprocessorOpts(CI.getPreprocessorOpts()),
        TargetOpts(CI.getTargetOpts()), LangOpts(CI.getLangOpts()), OS(OS),
        Buffer(Buffer) {
    // The debug info output isn't affected by CodeModel and
    // ThreadModel, but the backend expects them to be nonempty.
    CodeGenOpts.CodeModel = "default";
    CodeGenOpts.ThreadModel = "single";
    CodeGenOpts.DebugTypeExtRefs = true;
    CodeGenOpts.setDebugInfo(CodeGenOptions::FullDebugInfo);
  }

  ~PCHContainerGenerator() override = default;

  void Initialize(ASTContext &Context) override {
    assert(!Ctx && "initialized multiple times");

    Ctx = &Context;
    VMContext.reset(new llvm::LLVMContext());
    M.reset(new llvm::Module(MainFileName, *VMContext));
    M->setDataLayout(Ctx->getTargetInfo().getDataLayoutString());
    Builder.reset(new CodeGen::CodeGenModule(
        *Ctx, HeaderSearchOpts, PreprocessorOpts, CodeGenOpts, *M, Diags));
    Builder->getModuleDebugInfo()->setModuleMap(MMap);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    if (Diags.hasErrorOccurred())
      return true;

    // Collect debug info for all decls in this group.
    for (auto *I : D)
      if (!I->isFromASTFile()) {
        DebugTypeVisitor DTV(*Builder->getModuleDebugInfo(), *Ctx);
        DTV.TraverseDecl(I);
      }
    return true;
  }

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) override {
    HandleTopLevelDecl(D);
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    if (Diags.hasErrorOccurred())
      return;

    if (D->isFromASTFile())
      return;

    // Anonymous tag decls are deferred until we are building their declcontext.
    if (D->getName().empty())
      return;

    DebugTypeVisitor DTV(*Builder->getModuleDebugInfo(), *Ctx);
    DTV.TraverseDecl(D);
    Builder->UpdateCompletedType(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    if (Diags.hasErrorOccurred())
      return;

    if (const RecordDecl *RD = dyn_cast<RecordDecl>(D))
      Builder->getModuleDebugInfo()->completeRequiredType(RD);
  }

  /// Emit a container holding the serialized AST.
  void HandleTranslationUnit(ASTContext &Ctx) override {
    assert(M && VMContext && Builder);
    // Delete these on function exit.
    std::unique_ptr<llvm::LLVMContext> VMContext = std::move(this->VMContext);
    std::unique_ptr<llvm::Module> M = std::move(this->M);
    std::unique_ptr<CodeGen::CodeGenModule> Builder = std::move(this->Builder);

    if (Diags.hasErrorOccurred())
      return;

    M->setTargetTriple(Ctx.getTargetInfo().getTriple().getTriple());
    M->setDataLayout(Ctx.getTargetInfo().getDataLayoutString());

    // PCH files don't have a signature field in the control block,
    // but LLVM detects DWO CUs by looking for a non-zero DWO id.
    uint64_t Signature = Buffer->Signature ? Buffer->Signature : ~1UL;
    Builder->getModuleDebugInfo()->setDwoId(Signature);

    // Finalize the Builder.
    if (Builder)
      Builder->Release();

    // Ensure the target exists.
    std::string Error;
    auto Triple = Ctx.getTargetInfo().getTriple();
    if (!llvm::TargetRegistry::lookupTarget(Triple.getTriple(), Error))
      llvm::report_fatal_error(Error);

    // Emit the serialized Clang AST into its own section.
    assert(Buffer->IsComplete && "serialization did not complete");
    auto &SerializedAST = Buffer->Data;
    auto Size = SerializedAST.size();
    auto Int8Ty = llvm::Type::getInt8Ty(*VMContext);
    auto *Ty = llvm::ArrayType::get(Int8Ty, Size);
    auto *Data = llvm::ConstantDataArray::getString(
        *VMContext, StringRef(SerializedAST.data(), Size),
        /*AddNull=*/false);
    auto *ASTSym = new llvm::GlobalVariable(
        *M, Ty, /*constant*/ true, llvm::GlobalVariable::InternalLinkage, Data,
        "__clang_ast");
    // The on-disk hashtable needs to be aligned.
    ASTSym->setAlignment(8);

    // Mach-O also needs a segment name.
    if (Triple.isOSBinFormatMachO())
      ASTSym->setSection("__CLANG,__clangast");
    // COFF has an eight character length limit.
    else if (Triple.isOSBinFormatCOFF())
      ASTSym->setSection("clangast");
    else
      ASTSym->setSection("__clangast");

    DEBUG({
      // Print the IR for the PCH container to the debug output.
      llvm::SmallString<0> Buffer;
      llvm::raw_svector_ostream OS(Buffer);
      clang::EmitBackendOutput(Diags, CodeGenOpts, TargetOpts, LangOpts,
                               Ctx.getTargetInfo().getDataLayoutString(),
                               M.get(), BackendAction::Backend_EmitLL, &OS);
      llvm::dbgs() << Buffer;
    });

    // Use the LLVM backend to emit the pch container.
    clang::EmitBackendOutput(Diags, CodeGenOpts, TargetOpts, LangOpts,
                             Ctx.getTargetInfo().getDataLayoutString(),
                             M.get(), BackendAction::Backend_EmitObj, OS);

    // Make sure the pch container hits disk.
    OS->flush();

    // Free the memory for the temporary buffer.
    llvm::SmallVector<char, 0> Empty;
    SerializedAST = std::move(Empty);
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer>
ObjectFilePCHContainerWriter::CreatePCHContainerGenerator(
    CompilerInstance &CI, const std::string &MainFileName,
    const std::string &OutputFileName, llvm::raw_pwrite_stream *OS,
    std::shared_ptr<PCHBuffer> Buffer) const {
  return llvm::make_unique<PCHContainerGenerator>(CI, MainFileName,
                                                  OutputFileName, OS, Buffer);
}

void ObjectFilePCHContainerReader::ExtractPCH(
    llvm::MemoryBufferRef Buffer, llvm::BitstreamReader &StreamFile) const {
  if (auto OF = llvm::object::ObjectFile::createObjectFile(Buffer)) {
    auto *Obj = OF.get().get();
    bool IsCOFF = isa<llvm::object::COFFObjectFile>(Obj);
    // Find the clang AST section in the container.
    for (auto &Section : OF->get()->sections()) {
      StringRef Name;
      Section.getName(Name);
      if ((!IsCOFF && Name == "__clangast") ||
          ( IsCOFF && Name ==   "clangast")) {
        StringRef Buf;
        Section.getContents(Buf);
        StreamFile.init((const unsigned char *)Buf.begin(),
                        (const unsigned char *)Buf.end());
        return;
      }
    }
  }

  // As a fallback, treat the buffer as a raw AST.
  StreamFile.init((const unsigned char *)Buffer.getBufferStart(),
                  (const unsigned char *)Buffer.getBufferEnd());
}
