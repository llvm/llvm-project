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

public:
  PCHContainerGenerator(DiagnosticsEngine &diags,
                        const HeaderSearchOptions &HSO,
                        const PreprocessorOptions &PPO, const TargetOptions &TO,
                        const LangOptions &LO, const std::string &MainFileName,
                        const std::string &OutputFileName,
                        raw_pwrite_stream *OS,
                        std::shared_ptr<PCHBuffer> Buffer)
      : Diags(diags), HeaderSearchOpts(HSO), PreprocessorOpts(PPO),
        TargetOpts(TO), LangOpts(LO), OS(OS), Buffer(Buffer) {
    // The debug info output isn't affected by CodeModel and
    // ThreadModel, but the backend expects them to be nonempty.
    CodeGenOpts.CodeModel = "default";
    CodeGenOpts.ThreadModel = "single";
    CodeGenOpts.setDebugInfo(CodeGenOptions::FullDebugInfo);
    CodeGenOpts.SplitDwarfFile = OutputFileName;
  }

  virtual ~PCHContainerGenerator() {}

  void Initialize(ASTContext &Context) override {
    Ctx = &Context;
    VMContext.reset(new llvm::LLVMContext());
    M.reset(new llvm::Module(MainFileName, *VMContext));
    M->setDataLayout(Ctx->getTargetInfo().getTargetDescription());
    Builder.reset(new CodeGen::CodeGenModule(*Ctx, HeaderSearchOpts,
                                             PreprocessorOpts, CodeGenOpts, *M,
                                             M->getDataLayout(), Diags));
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
    M->setDataLayout(Ctx.getTargetInfo().getTargetDescription());

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
                               Ctx.getTargetInfo().getTargetDescription(),
                               M.get(), BackendAction::Backend_EmitLL, &OS);
      OS.flush();
      llvm::dbgs() << Buffer;
    });

    // Use the LLVM backend to emit the pch container.
    clang::EmitBackendOutput(Diags, CodeGenOpts, TargetOpts, LangOpts,
                             Ctx.getTargetInfo().getTargetDescription(),
                             M.get(), BackendAction::Backend_EmitObj, OS);

    // Make sure the pch container hits disk.
    OS->flush();

    // Free the memory for the temporary buffer.
    llvm::SmallVector<char, 0> Empty;
    SerializedAST = std::move(Empty);
  }
};

} // namespace

std::unique_ptr<ASTConsumer>
ObjectFilePCHContainerWriter::CreatePCHContainerGenerator(
    DiagnosticsEngine &Diags, const HeaderSearchOptions &HSO,
    const PreprocessorOptions &PPO, const TargetOptions &TO,
    const LangOptions &LO, const std::string &MainFileName,
    const std::string &OutputFileName, llvm::raw_pwrite_stream *OS,
    std::shared_ptr<PCHBuffer> Buffer) const {
  return llvm::make_unique<PCHContainerGenerator>(
      Diags, HSO, PPO, TO, LO, MainFileName, OutputFileName, OS, Buffer);
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
  return;
}
