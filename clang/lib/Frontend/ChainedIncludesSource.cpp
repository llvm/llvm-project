//===- ChainedIncludesSource.cpp - Chained PCHs in Memory -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ChainedIncludesSource class, which converts headers
//  to chained PCHs in memory, mainly used for testing.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace clang;

namespace {
class ChainedIncludesSource : public ExternalSemaSource {
public:
  ~ChainedIncludesSource() override;

  ExternalSemaSource &getFinalReader() const { return *FinalReader; }

  std::vector<CompilerInstance *> CIs;
  IntrusiveRefCntPtr<ExternalSemaSource> FinalReader;

protected:
  //===----------------------------------------------------------------------===//
  // ExternalASTSource interface.
  //===----------------------------------------------------------------------===//

  Decl *GetExternalDecl(uint32_t ID) override;
  Selector GetExternalSelector(uint32_t ID) override;
  uint32_t GetNumExternalSelectors() override;
  Stmt *GetExternalDeclStmt(uint64_t Offset) override;
  CXXCtorInitializer **GetExternalCXXCtorInitializers(uint64_t Offset) override;
  CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset) override;
  bool FindExternalVisibleDeclsByName(const DeclContext *DC,
                                      DeclarationName Name) override;
  void
  FindExternalLexicalDecls(const DeclContext *DC,
                           llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
                           SmallVectorImpl<Decl *> &Result) override;
  void CompleteType(TagDecl *Tag) override;
  void CompleteType(ObjCInterfaceDecl *Class) override;
  void StartedDeserializing() override;
  void FinishedDeserializing() override;
  void StartTranslationUnit(ASTConsumer *Consumer) override;
  void PrintStats() override;

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  void getMemoryBufferSizes(MemoryBufferSizes &sizes) const override;

  //===----------------------------------------------------------------------===//
  // ExternalSemaSource interface.
  //===----------------------------------------------------------------------===//

  void InitializeSema(Sema &S) override;
  void ForgetSema() override;
  void ReadMethodPool(Selector Sel) override;
  bool LookupUnqualified(LookupResult &R, Scope *S) override;
};
}

static ASTReader *
createASTReader(CompilerInstance &CI, StringRef pchFile,
                SmallVectorImpl<std::unique_ptr<llvm::MemoryBuffer>> &MemBufs,
                SmallVectorImpl<std::string> &bufNames,
                ASTDeserializationListener *deserialListener = nullptr) {
  Preprocessor &PP = CI.getPreprocessor();
  std::unique_ptr<ASTReader> Reader;
  Reader.reset(new ASTReader(PP, CI.getASTContext(),
                             CI.getPCHContainerReader(),
                             /*Extensions=*/{ },
                             /*isysroot=*/"", /*DisableValidation=*/true));
  for (unsigned ti = 0; ti < bufNames.size(); ++ti) {
    StringRef sr(bufNames[ti]);
    Reader->addInMemoryBuffer(sr, std::move(MemBufs[ti]));
  }
  Reader->setDeserializationListener(deserialListener);
  switch (Reader->ReadAST(pchFile, serialization::MK_PCH, SourceLocation(),
                          ASTReader::ARR_None)) {
  case ASTReader::Success:
    // Set the predefines buffer as suggested by the PCH reader.
    PP.setPredefines(Reader->getSuggestedPredefines());
    return Reader.release();

  case ASTReader::Failure:
  case ASTReader::Missing:
  case ASTReader::OutOfDate:
  case ASTReader::VersionMismatch:
  case ASTReader::ConfigurationMismatch:
  case ASTReader::HadErrors:
    break;
  }
  return nullptr;
}

ChainedIncludesSource::~ChainedIncludesSource() {
  for (unsigned i = 0, e = CIs.size(); i != e; ++i)
    delete CIs[i];
}

IntrusiveRefCntPtr<ExternalSemaSource> clang::createChainedIncludesSource(
    CompilerInstance &CI, IntrusiveRefCntPtr<ExternalSemaSource> &Reader) {

  std::vector<std::string> &includes = CI.getPreprocessorOpts().ChainedIncludes;
  assert(!includes.empty() && "No '-chain-include' in options!");

  IntrusiveRefCntPtr<ChainedIncludesSource> source(new ChainedIncludesSource());
  InputKind IK = CI.getFrontendOpts().Inputs[0].getKind();

  SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 4> SerialBufs;
  SmallVector<std::string, 4> serialBufNames;

  for (unsigned i = 0, e = includes.size(); i != e; ++i) {
    bool firstInclude = (i == 0);
    std::unique_ptr<CompilerInvocation> CInvok;
    CInvok.reset(new CompilerInvocation(CI.getInvocation()));
    
    CInvok->getPreprocessorOpts().ChainedIncludes.clear();
    CInvok->getPreprocessorOpts().ImplicitPCHInclude.clear();
    CInvok->getPreprocessorOpts().ImplicitPTHInclude.clear();
    CInvok->getPreprocessorOpts().DisablePCHValidation = true;
    CInvok->getPreprocessorOpts().Includes.clear();
    CInvok->getPreprocessorOpts().MacroIncludes.clear();
    CInvok->getPreprocessorOpts().Macros.clear();
    
    CInvok->getFrontendOpts().Inputs.clear();
    FrontendInputFile InputFile(includes[i], IK);
    CInvok->getFrontendOpts().Inputs.push_back(InputFile);

    TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), new DiagnosticOptions());
    IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
        new DiagnosticsEngine(DiagID, &CI.getDiagnosticOpts(), DiagClient));

    std::unique_ptr<CompilerInstance> Clang(
        new CompilerInstance(CI.getPCHContainerOperations()));
    Clang->setInvocation(CInvok.release());
    Clang->setDiagnostics(Diags.get());
    Clang->setTarget(TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
    Clang->createFileManager();
    Clang->createSourceManager(Clang->getFileManager());
    Clang->createPreprocessor(TU_Prefix);
    Clang->getDiagnosticClient().BeginSourceFile(Clang->getLangOpts(),
                                                 &Clang->getPreprocessor());
    Clang->createASTContext();

    auto Buffer = std::make_shared<PCHBuffer>();
    ArrayRef<llvm::IntrusiveRefCntPtr<ModuleFileExtension>> Extensions;
    auto consumer = llvm::make_unique<PCHGenerator>(
        Clang->getPreprocessor(), "-", nullptr, /*isysroot=*/"", Buffer,
        Extensions, /*AllowASTWithErrors=*/true);
    Clang->getASTContext().setASTMutationListener(
                                            consumer->GetASTMutationListener());
    Clang->setASTConsumer(std::move(consumer));
    Clang->createSema(TU_Prefix, nullptr);

    if (firstInclude) {
      Preprocessor &PP = Clang->getPreprocessor();
      PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(),
                                             PP.getLangOpts());
    } else {
      assert(!SerialBufs.empty());
      SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 4> Bufs;
      // TODO: Pass through the existing MemoryBuffer instances instead of
      // allocating new ones.
      for (auto &SB : SerialBufs)
        Bufs.push_back(llvm::MemoryBuffer::getMemBuffer(SB->getBuffer()));
      std::string pchName = includes[i-1];
      llvm::raw_string_ostream os(pchName);
      os << ".pch" << i-1;
      serialBufNames.push_back(os.str());

      IntrusiveRefCntPtr<ASTReader> Reader;
      Reader = createASTReader(
          *Clang, pchName, Bufs, serialBufNames,
          Clang->getASTConsumer().GetASTDeserializationListener());
      if (!Reader)
        return nullptr;
      Clang->setModuleManager(Reader);
      Clang->getASTContext().setExternalSource(Reader);
    }
    
    if (!Clang->InitializeSourceManager(InputFile))
      return nullptr;

    ParseAST(Clang->getSema());
    Clang->getDiagnosticClient().EndSourceFile();
    assert(Buffer->IsComplete && "serialization did not complete");
    auto &serialAST = Buffer->Data;
    SerialBufs.push_back(llvm::MemoryBuffer::getMemBufferCopy(
        StringRef(serialAST.data(), serialAST.size())));
    serialAST.clear();
    source->CIs.push_back(Clang.release());
  }

  assert(!SerialBufs.empty());
  std::string pchName = includes.back() + ".pch-final";
  serialBufNames.push_back(pchName);
  Reader = createASTReader(CI, pchName, SerialBufs, serialBufNames);
  if (!Reader)
    return nullptr;

  source->FinalReader = Reader;
  return source;
}

//===----------------------------------------------------------------------===//
// ExternalASTSource interface.
//===----------------------------------------------------------------------===//

Decl *ChainedIncludesSource::GetExternalDecl(uint32_t ID) {
  return getFinalReader().GetExternalDecl(ID);
}
Selector ChainedIncludesSource::GetExternalSelector(uint32_t ID) {
  return getFinalReader().GetExternalSelector(ID);
}
uint32_t ChainedIncludesSource::GetNumExternalSelectors() {
  return getFinalReader().GetNumExternalSelectors();
}
Stmt *ChainedIncludesSource::GetExternalDeclStmt(uint64_t Offset) {
  return getFinalReader().GetExternalDeclStmt(Offset);
}
CXXBaseSpecifier *
ChainedIncludesSource::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  return getFinalReader().GetExternalCXXBaseSpecifiers(Offset);
}
CXXCtorInitializer **
ChainedIncludesSource::GetExternalCXXCtorInitializers(uint64_t Offset) {
  return getFinalReader().GetExternalCXXCtorInitializers(Offset);
}
bool
ChainedIncludesSource::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                                      DeclarationName Name) {
  return getFinalReader().FindExternalVisibleDeclsByName(DC, Name);
}
void ChainedIncludesSource::FindExternalLexicalDecls(
    const DeclContext *DC, llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
    SmallVectorImpl<Decl *> &Result) {
  return getFinalReader().FindExternalLexicalDecls(DC, IsKindWeWant, Result);
}
void ChainedIncludesSource::CompleteType(TagDecl *Tag) {
  return getFinalReader().CompleteType(Tag);
}
void ChainedIncludesSource::CompleteType(ObjCInterfaceDecl *Class) {
  return getFinalReader().CompleteType(Class);
}
void ChainedIncludesSource::StartedDeserializing() {
  return getFinalReader().StartedDeserializing();
}
void ChainedIncludesSource::FinishedDeserializing() {
  return getFinalReader().FinishedDeserializing();
}
void ChainedIncludesSource::StartTranslationUnit(ASTConsumer *Consumer) {
  return getFinalReader().StartTranslationUnit(Consumer);
}
void ChainedIncludesSource::PrintStats() {
  return getFinalReader().PrintStats();
}
void ChainedIncludesSource::getMemoryBufferSizes(MemoryBufferSizes &sizes)const{
  for (unsigned i = 0, e = CIs.size(); i != e; ++i) {
    if (const ExternalASTSource *eSrc =
        CIs[i]->getASTContext().getExternalSource()) {
      eSrc->getMemoryBufferSizes(sizes);
    }
  }

  getFinalReader().getMemoryBufferSizes(sizes);
}

void ChainedIncludesSource::InitializeSema(Sema &S) {
  return getFinalReader().InitializeSema(S);
}
void ChainedIncludesSource::ForgetSema() {
  return getFinalReader().ForgetSema();
}
void ChainedIncludesSource::ReadMethodPool(Selector Sel) {
  getFinalReader().ReadMethodPool(Sel);
}
bool ChainedIncludesSource::LookupUnqualified(LookupResult &R, Scope *S) {
  return getFinalReader().LookupUnqualified(R, S);
}

