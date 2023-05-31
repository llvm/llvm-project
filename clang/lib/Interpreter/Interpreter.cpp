//===------ Interpreter.cpp - Incremental Compilation and Execution -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the component which performs incremental code
// compilation and execution.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/Interpreter.h"

#include "DeviceOffload.h"
#include "IncrementalExecutor.h"
#include "IncrementalParser.h"

#include "InterpreterUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Interpreter/Value.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Lookup.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
using namespace clang;

// FIXME: Figure out how to unify with namespace init_convenience from
//        tools/clang-import-test/clang-import-test.cpp
namespace {
/// Retrieves the clang CC1 specific flags out of the compilation's jobs.
/// \returns NULL on error.
static llvm::Expected<const llvm::opt::ArgStringList *>
GetCC1Arguments(DiagnosticsEngine *Diagnostics,
                driver::Compilation *Compilation) {
  // We expect to get back exactly one Command job, if we didn't something
  // failed. Extract that job from the Compilation.
  const driver::JobList &Jobs = Compilation->getJobs();
  if (!Jobs.size() || !isa<driver::Command>(*Jobs.begin()))
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Driver initialization failed. "
                                   "Unable to create a driver job");

  // The one job we find should be to invoke clang again.
  const driver::Command *Cmd = cast<driver::Command>(&(*Jobs.begin()));
  if (llvm::StringRef(Cmd->getCreator().getName()) != "clang")
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Driver initialization failed");

  return &Cmd->getArguments();
}

static llvm::Expected<std::unique_ptr<CompilerInstance>>
CreateCI(const llvm::opt::ArgStringList &Argv) {
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  // FIXME: Clang should register these container operations automatically.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
  bool Success = CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), llvm::ArrayRef(Argv.begin(), Argv.size()), Diags);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(Argv[0], nullptr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics())
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Unable to create diagnostics engine");

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Unable to flush diagnostics");

  // FIXME: Merge with CompilerInstance::ExecuteAction.
  llvm::MemoryBuffer *MB = llvm::MemoryBuffer::getMemBuffer("").release();
  Clang->getPreprocessorOpts().addRemappedFile("<<< inputs >>>", MB);

  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Target is missing");

  Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

  // Don't clear the AST before backend codegen since we do codegen multiple
  // times, reusing the same AST.
  Clang->getCodeGenOpts().ClearASTBeforeBackend = false;

  Clang->getFrontendOpts().DisableFree = false;
  Clang->getCodeGenOpts().DisableFree = false;

  return std::move(Clang);
}

} // anonymous namespace

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::create(std::vector<const char *> &ClangArgv) {

  // If we don't know ClangArgv0 or the address of main() at this point, try
  // to guess it anyway (it's possible on some platforms).
  std::string MainExecutableName =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);

  ClangArgv.insert(ClangArgv.begin(), MainExecutableName.c_str());

  // Prepending -c to force the driver to do something if no action was
  // specified. By prepending we allow users to override the default
  // action and use other actions in incremental mode.
  // FIXME: Print proper driver diagnostics if the driver flags are wrong.
  // We do C++ by default; append right after argv[0] if no "-x" given
  ClangArgv.insert(ClangArgv.end(), "-Xclang");
  ClangArgv.insert(ClangArgv.end(), "-fincremental-extensions");
  ClangArgv.insert(ClangArgv.end(), "-c");

  // Put a dummy C++ file on to ensure there's at least one compile job for the
  // driver to construct.
  ClangArgv.push_back("<<< inputs >>>");

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(ClangArgv);
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  driver::Driver Driver(/*MainBinaryName=*/ClangArgv[0],
                        llvm::sys::getProcessTriple(), Diags);
  Driver.setCheckInputsExist(false); // the input comes from mem buffers
  llvm::ArrayRef<const char *> RF = llvm::ArrayRef(ClangArgv);
  std::unique_ptr<driver::Compilation> Compilation(Driver.BuildCompilation(RF));

  if (Compilation->getArgs().hasArg(driver::options::OPT_v))
    Compilation->getJobs().Print(llvm::errs(), "\n", /*Quote=*/false);

  auto ErrOrCC1Args = GetCC1Arguments(&Diags, Compilation.get());
  if (auto Err = ErrOrCC1Args.takeError())
    return std::move(Err);

  return CreateCI(**ErrOrCC1Args);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCpp() {
  std::vector<const char *> Argv;
  Argv.reserve(5 + 1 + UserArgs.size());
  Argv.push_back("-xc++");
  Argv.insert(Argv.end(), UserArgs.begin(), UserArgs.end());

  return IncrementalCompilerBuilder::create(Argv);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::createCuda(bool device) {
  std::vector<const char *> Argv;
  Argv.reserve(5 + 4 + UserArgs.size());

  Argv.push_back("-xcuda");
  if (device)
    Argv.push_back("--cuda-device-only");
  else
    Argv.push_back("--cuda-host-only");

  std::string SDKPathArg = "--cuda-path=";
  if (!CudaSDKPath.empty()) {
    SDKPathArg += CudaSDKPath;
    Argv.push_back(SDKPathArg.c_str());
  }

  std::string ArchArg = "--offload-arch=";
  if (!OffloadArch.empty()) {
    ArchArg += OffloadArch;
    Argv.push_back(ArchArg.c_str());
  }

  Argv.insert(Argv.end(), UserArgs.begin(), UserArgs.end());

  return IncrementalCompilerBuilder::create(Argv);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCudaDevice() {
  return IncrementalCompilerBuilder::createCuda(true);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCudaHost() {
  return IncrementalCompilerBuilder::createCuda(false);
}

Interpreter::Interpreter(std::unique_ptr<CompilerInstance> CI,
                         llvm::Error &Err) {
  llvm::ErrorAsOutParameter EAO(&Err);
  auto LLVMCtx = std::make_unique<llvm::LLVMContext>();
  TSCtx = std::make_unique<llvm::orc::ThreadSafeContext>(std::move(LLVMCtx));
  IncrParser = std::make_unique<IncrementalParser>(*this, std::move(CI),
                                                   *TSCtx->getContext(), Err);
}

Interpreter::~Interpreter() {
  if (IncrExecutor) {
    if (llvm::Error Err = IncrExecutor->cleanUp())
      llvm::report_fatal_error(
          llvm::Twine("Failed to clean up IncrementalExecutor: ") +
          toString(std::move(Err)));
  }
}

// These better to put in a runtime header but we can't. This is because we
// can't find the precise resource directory in unittests so we have to hard
// code them.
const char *const Runtimes = R"(
    void* operator new(__SIZE_TYPE__, void* __p) noexcept;
    void *__clang_Interpreter_SetValueWithAlloc(void*, void*, void*);
    void __clang_Interpreter_SetValueNoAlloc(void*, void*, void*);
    void __clang_Interpreter_SetValueNoAlloc(void*, void*, void*, void*);
    void __clang_Interpreter_SetValueNoAlloc(void*, void*, void*, float);
    void __clang_Interpreter_SetValueNoAlloc(void*, void*, void*, double);
    void __clang_Interpreter_SetValueNoAlloc(void*, void*, void*, long double);
    void __clang_Interpreter_SetValueNoAlloc(void*,void*,void*,unsigned long long);
    template <class T, class = T (*)() /*disable for arrays*/>
    void __clang_Interpreter_SetValueCopyArr(T* Src, void* Placement, unsigned long Size) {
      for (auto Idx = 0; Idx < Size; ++Idx)
        new ((void*)(((T*)Placement) + Idx)) T(Src[Idx]);
    }
    template <class T, unsigned long N>
    void __clang_Interpreter_SetValueCopyArr(const T (*Src)[N], void* Placement, unsigned long Size) {
      __clang_Interpreter_SetValueCopyArr(Src[0], Placement, Size);
    }
)";

llvm::Expected<std::unique_ptr<Interpreter>>
Interpreter::create(std::unique_ptr<CompilerInstance> CI) {
  llvm::Error Err = llvm::Error::success();
  auto Interp =
      std::unique_ptr<Interpreter>(new Interpreter(std::move(CI), Err));
  if (Err)
    return std::move(Err);
  auto PTU = Interp->Parse(Runtimes);
  if (!PTU)
    return PTU.takeError();

  Interp->ValuePrintingInfo.resize(3);
  // FIXME: This is a ugly hack. Undo command checks its availability by looking
  // at the size of the PTU list. However we have parsed something in the
  // beginning of the REPL so we have to mark them as 'Irrevocable'.
  Interp->InitPTUSize = Interp->IncrParser->getPTUs().size();
  return std::move(Interp);
}

llvm::Expected<std::unique_ptr<Interpreter>>
Interpreter::createWithCUDA(std::unique_ptr<CompilerInstance> CI,
                            std::unique_ptr<CompilerInstance> DCI) {
  // avoid writing fat binary to disk using an in-memory virtual file system
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> IMVFS =
      std::make_unique<llvm::vfs::InMemoryFileSystem>();
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayVFS =
      std::make_unique<llvm::vfs::OverlayFileSystem>(
          llvm::vfs::getRealFileSystem());
  OverlayVFS->pushOverlay(IMVFS);
  CI->createFileManager(OverlayVFS);

  auto Interp = Interpreter::create(std::move(CI));
  if (auto E = Interp.takeError())
    return std::move(E);

  llvm::Error Err = llvm::Error::success();
  auto DeviceParser = std::make_unique<IncrementalCUDADeviceParser>(
      **Interp, std::move(DCI), *(*Interp)->IncrParser.get(),
      *(*Interp)->TSCtx->getContext(), IMVFS, Err);
  if (Err)
    return std::move(Err);

  (*Interp)->DeviceParser = std::move(DeviceParser);

  return Interp;
}

const CompilerInstance *Interpreter::getCompilerInstance() const {
  return IncrParser->getCI();
}

llvm::Expected<llvm::orc::LLJIT &> Interpreter::getExecutionEngine() {
  if (!IncrExecutor) {
    if (auto Err = CreateExecutor())
      return std::move(Err);
  }

  return IncrExecutor->GetExecutionEngine();
}

ASTContext &Interpreter::getASTContext() {
  return getCompilerInstance()->getASTContext();
}

const ASTContext &Interpreter::getASTContext() const {
  return getCompilerInstance()->getASTContext();
}

size_t Interpreter::getEffectivePTUSize() const {
  std::list<PartialTranslationUnit> &PTUs = IncrParser->getPTUs();
  assert(PTUs.size() >= InitPTUSize && "empty PTU list?");
  return PTUs.size() - InitPTUSize;
}

llvm::Expected<PartialTranslationUnit &>
Interpreter::Parse(llvm::StringRef Code) {
  // If we have a device parser, parse it first.
  // The generated code will be included in the host compilation
  if (DeviceParser) {
    auto DevicePTU = DeviceParser->Parse(Code);
    if (auto E = DevicePTU.takeError())
      return std::move(E);
  }

  // Tell the interpreter sliently ignore unused expressions since value
  // printing could cause it.
  getCompilerInstance()->getDiagnostics().setSeverity(
      clang::diag::warn_unused_expr, diag::Severity::Ignored, SourceLocation());
  return IncrParser->Parse(Code);
}

llvm::Error Interpreter::CreateExecutor() {
  const clang::TargetInfo &TI =
      getCompilerInstance()->getASTContext().getTargetInfo();
  llvm::Error Err = llvm::Error::success();
  auto Executor = std::make_unique<IncrementalExecutor>(*TSCtx, Err, TI);
  if (!Err)
    IncrExecutor = std::move(Executor);

  return Err;
}

llvm::Error Interpreter::Execute(PartialTranslationUnit &T) {
  assert(T.TheModule);
  if (!IncrExecutor) {
    auto Err = CreateExecutor();
    if (Err)
      return Err;
  }
  // FIXME: Add a callback to retain the llvm::Module once the JIT is done.
  if (auto Err = IncrExecutor->addModule(T))
    return Err;

  if (auto Err = IncrExecutor->runCtors())
    return Err;

  return llvm::Error::success();
}

llvm::Error Interpreter::ParseAndExecute(llvm::StringRef Code, Value *V) {

  auto PTU = Parse(Code);
  if (!PTU)
    return PTU.takeError();
  if (PTU->TheModule)
    if (llvm::Error Err = Execute(*PTU))
      return Err;

  if (LastValue.isValid()) {
    if (!V) {
      LastValue.dump();
      LastValue.clear();
    } else
      *V = std::move(LastValue);
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddress(GlobalDecl GD) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());
  llvm::StringRef MangledName = IncrParser->GetMangledName(GD);
  return getSymbolAddress(MangledName);
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddress(llvm::StringRef IRName) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());

  return IncrExecutor->getSymbolAddress(IRName, IncrementalExecutor::IRName);
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddressFromLinkerName(llvm::StringRef Name) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());

  return IncrExecutor->getSymbolAddress(Name, IncrementalExecutor::LinkerName);
}

llvm::Error Interpreter::Undo(unsigned N) {

  std::list<PartialTranslationUnit> &PTUs = IncrParser->getPTUs();
  if (N > getEffectivePTUSize())
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "Too many undos",
                                               std::error_code());
  for (unsigned I = 0; I < N; I++) {
    if (IncrExecutor) {
      if (llvm::Error Err = IncrExecutor->removeModule(PTUs.back()))
        return Err;
    }

    IncrParser->CleanUpPTU(PTUs.back());
    PTUs.pop_back();
  }
  return llvm::Error::success();
}

llvm::Error Interpreter::LoadDynamicLibrary(const char *name) {
  auto EE = getExecutionEngine();
  if (!EE)
    return EE.takeError();

  auto &DL = EE->getDataLayout();

  if (auto DLSG = llvm::orc::DynamicLibrarySearchGenerator::Load(
          name, DL.getGlobalPrefix()))
    EE->getMainJITDylib().addGenerator(std::move(*DLSG));
  else
    return DLSG.takeError();

  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::CompileDtorCall(CXXRecordDecl *CXXRD) {
  assert(CXXRD && "Cannot compile a destructor for a nullptr");
  if (auto Dtor = Dtors.find(CXXRD); Dtor != Dtors.end())
    return Dtor->getSecond();

  if (CXXRD->hasIrrelevantDestructor())
    return llvm::orc::ExecutorAddr{};

  CXXDestructorDecl *DtorRD =
      getCompilerInstance()->getSema().LookupDestructor(CXXRD);

  llvm::StringRef Name =
      IncrParser->GetMangledName(GlobalDecl(DtorRD, Dtor_Base));
  auto AddrOrErr = getSymbolAddress(Name);
  if (!AddrOrErr)
    return AddrOrErr.takeError();

  Dtors[CXXRD] = *AddrOrErr;
  return AddrOrErr;
}

static constexpr llvm::StringRef MagicRuntimeInterface[] = {
    "__clang_Interpreter_SetValueNoAlloc",
    "__clang_Interpreter_SetValueWithAlloc",
    "__clang_Interpreter_SetValueCopyArr"};

bool Interpreter::FindRuntimeInterface() {
  if (llvm::all_of(ValuePrintingInfo, [](Expr *E) { return E != nullptr; }))
    return true;

  Sema &S = getCompilerInstance()->getSema();
  ASTContext &Ctx = S.getASTContext();

  auto LookupInterface = [&](Expr *&Interface, llvm::StringRef Name) {
    LookupResult R(S, &Ctx.Idents.get(Name), SourceLocation(),
                   Sema::LookupOrdinaryName, Sema::ForVisibleRedeclaration);
    S.LookupQualifiedName(R, Ctx.getTranslationUnitDecl());
    if (R.empty())
      return false;

    CXXScopeSpec CSS;
    Interface = S.BuildDeclarationNameExpr(CSS, R, /*ADL=*/false).get();
    return true;
  };

  if (!LookupInterface(ValuePrintingInfo[NoAlloc],
                       MagicRuntimeInterface[NoAlloc]))
    return false;
  if (!LookupInterface(ValuePrintingInfo[WithAlloc],
                       MagicRuntimeInterface[WithAlloc]))
    return false;
  if (!LookupInterface(ValuePrintingInfo[CopyArray],
                       MagicRuntimeInterface[CopyArray]))
    return false;
  return true;
}

namespace {

class RuntimeInterfaceBuilder
    : public TypeVisitor<RuntimeInterfaceBuilder, Interpreter::InterfaceKind> {
  clang::Interpreter &Interp;
  ASTContext &Ctx;
  Sema &S;
  Expr *E;
  llvm::SmallVector<Expr *, 3> Args;

public:
  RuntimeInterfaceBuilder(clang::Interpreter &In, ASTContext &C, Sema &SemaRef,
                          Expr *VE, ArrayRef<Expr *> FixedArgs)
      : Interp(In), Ctx(C), S(SemaRef), E(VE) {
    // The Interpreter* parameter and the out parameter `OutVal`.
    for (Expr *E : FixedArgs)
      Args.push_back(E);

    // Get rid of ExprWithCleanups.
    if (auto *EWC = llvm::dyn_cast_if_present<ExprWithCleanups>(E))
      E = EWC->getSubExpr();
  }

  ExprResult getCall() {
    QualType Ty = E->getType();
    QualType DesugaredTy = Ty.getDesugaredType(Ctx);

    // For lvalue struct, we treat it as a reference.
    if (DesugaredTy->isRecordType() && E->isLValue()) {
      DesugaredTy = Ctx.getLValueReferenceType(DesugaredTy);
      Ty = Ctx.getLValueReferenceType(Ty);
    }

    Expr *TypeArg =
        CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)Ty.getAsOpaquePtr());
    // The QualType parameter `OpaqueType`, represented as `void*`.
    Args.push_back(TypeArg);

    // We push the last parameter based on the type of the Expr. Note we need
    // special care for rvalue struct.
    Interpreter::InterfaceKind Kind = Visit(&*DesugaredTy);
    switch (Kind) {
    case Interpreter::InterfaceKind::WithAlloc:
    case Interpreter::InterfaceKind::CopyArray: {
      // __clang_Interpreter_SetValueWithAlloc.
      ExprResult AllocCall = S.ActOnCallExpr(
          /*Scope=*/nullptr,
          Interp.getValuePrintingInfo()[Interpreter::InterfaceKind::WithAlloc],
          E->getBeginLoc(), Args, E->getEndLoc());
      assert(!AllocCall.isInvalid() && "Can't create runtime interface call!");

      TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());

      // Force CodeGen to emit destructor.
      if (auto *RD = Ty->getAsCXXRecordDecl()) {
        auto *Dtor = S.LookupDestructor(RD);
        Dtor->addAttr(UsedAttr::CreateImplicit(Ctx));
        Interp.getCompilerInstance()->getASTConsumer().HandleTopLevelDecl(
            DeclGroupRef(Dtor));
      }

      // __clang_Interpreter_SetValueCopyArr.
      if (Kind == Interpreter::InterfaceKind::CopyArray) {
        const auto *ConstantArrTy =
            cast<ConstantArrayType>(DesugaredTy.getTypePtr());
        size_t ArrSize = Ctx.getConstantArrayElementCount(ConstantArrTy);
        Expr *ArrSizeExpr = IntegerLiteralExpr(Ctx, ArrSize);
        Expr *Args[] = {E, AllocCall.get(), ArrSizeExpr};
        return S.ActOnCallExpr(
            /*Scope *=*/nullptr,
            Interp
                .getValuePrintingInfo()[Interpreter::InterfaceKind::CopyArray],
            SourceLocation(), Args, SourceLocation());
      }
      Expr *Args[] = {AllocCall.get()};
      ExprResult CXXNewCall = S.BuildCXXNew(
          E->getSourceRange(),
          /*UseGlobal=*/true, /*PlacementLParen=*/SourceLocation(), Args,
          /*PlacementRParen=*/SourceLocation(),
          /*TypeIdParens=*/SourceRange(), TSI->getType(), TSI, std::nullopt,
          E->getSourceRange(), E);

      assert(!CXXNewCall.isInvalid() &&
             "Can't create runtime placement new call!");

      return S.ActOnFinishFullExpr(CXXNewCall.get(),
                                   /*DiscardedValue=*/false);
    }
      // __clang_Interpreter_SetValueNoAlloc.
    case Interpreter::InterfaceKind::NoAlloc: {
      return S.ActOnCallExpr(
          /*Scope=*/nullptr,
          Interp.getValuePrintingInfo()[Interpreter::InterfaceKind::NoAlloc],
          E->getBeginLoc(), Args, E->getEndLoc());
    }
    }
    llvm_unreachable("Unhandled Interpreter::InterfaceKind");
  }

  Interpreter::InterfaceKind VisitRecordType(const RecordType *Ty) {
    return Interpreter::InterfaceKind::WithAlloc;
  }

  Interpreter::InterfaceKind
  VisitMemberPointerType(const MemberPointerType *Ty) {
    return Interpreter::InterfaceKind::WithAlloc;
  }

  Interpreter::InterfaceKind
  VisitConstantArrayType(const ConstantArrayType *Ty) {
    return Interpreter::InterfaceKind::CopyArray;
  }

  Interpreter::InterfaceKind
  VisitFunctionProtoType(const FunctionProtoType *Ty) {
    HandlePtrType(Ty);
    return Interpreter::InterfaceKind::NoAlloc;
  }

  Interpreter::InterfaceKind VisitPointerType(const PointerType *Ty) {
    HandlePtrType(Ty);
    return Interpreter::InterfaceKind::NoAlloc;
  }

  Interpreter::InterfaceKind VisitReferenceType(const ReferenceType *Ty) {
    ExprResult AddrOfE = S.CreateBuiltinUnaryOp(SourceLocation(), UO_AddrOf, E);
    assert(!AddrOfE.isInvalid() && "Can not create unary expression");
    Args.push_back(AddrOfE.get());
    return Interpreter::InterfaceKind::NoAlloc;
  }

  Interpreter::InterfaceKind VisitBuiltinType(const BuiltinType *Ty) {
    if (Ty->isNullPtrType())
      Args.push_back(E);
    else if (Ty->isFloatingType())
      Args.push_back(E);
    else if (Ty->isIntegralOrEnumerationType())
      HandleIntegralOrEnumType(Ty);
    else if (Ty->isVoidType()) {
      // Do we need to still run `E`?
    }

    return Interpreter::InterfaceKind::NoAlloc;
  }

  Interpreter::InterfaceKind VisitEnumType(const EnumType *Ty) {
    HandleIntegralOrEnumType(Ty);
    return Interpreter::InterfaceKind::NoAlloc;
  }

private:
  // Force cast these types to uint64 to reduce the number of overloads of
  // `__clang_Interpreter_SetValueNoAlloc`.
  void HandleIntegralOrEnumType(const Type *Ty) {
    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ctx.UnsignedLongLongTy);
    ExprResult CastedExpr =
        S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E);
    assert(!CastedExpr.isInvalid() && "Cannot create cstyle cast expr");
    Args.push_back(CastedExpr.get());
  }

  void HandlePtrType(const Type *Ty) {
    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ctx.VoidPtrTy);
    ExprResult CastedExpr =
        S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E);
    assert(!CastedExpr.isInvalid() && "Can not create cstyle cast expression");
    Args.push_back(CastedExpr.get());
  }
};
} // namespace

// This synthesizes a call expression to a speciall
// function that is responsible for generating the Value.
// In general, we transform:
//   clang-repl> x
// To:
//   // 1. If x is a built-in type like int, float.
//   __clang_Interpreter_SetValueNoAlloc(ThisInterp, OpaqueValue, xQualType, x);
//   // 2. If x is a struct, and a lvalue.
//   __clang_Interpreter_SetValueNoAlloc(ThisInterp, OpaqueValue, xQualType,
//   &x);
//   // 3. If x is a struct, but a rvalue.
//   new (__clang_Interpreter_SetValueWithAlloc(ThisInterp, OpaqueValue,
//   xQualType)) (x);

Expr *Interpreter::SynthesizeExpr(Expr *E) {
  Sema &S = getCompilerInstance()->getSema();
  ASTContext &Ctx = S.getASTContext();

  if (!FindRuntimeInterface())
    llvm_unreachable("We can't find the runtime iterface for pretty print!");

  // Create parameter `ThisInterp`.
  auto *ThisInterp = CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)this);

  // Create parameter `OutVal`.
  auto *OutValue = CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)&LastValue);

  // Build `__clang_Interpreter_SetValue*` call.
  RuntimeInterfaceBuilder Builder(*this, Ctx, S, E, {ThisInterp, OutValue});

  ExprResult Result = Builder.getCall();
  // It could fail, like printing an array type in C. (not supported)
  if (Result.isInvalid())
    return E;
  return Result.get();
}

// Temporary rvalue struct that need special care.
REPL_EXTERNAL_VISIBILITY void *
__clang_Interpreter_SetValueWithAlloc(void *This, void *OutVal,
                                      void *OpaqueType) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  return VRef.getPtr();
}

// Pointers, lvalue struct that can take as a reference.
REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType,
                                    void *Val) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  VRef.setPtr(Val);
}

REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal,
                                    void *OpaqueType) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
}

static void SetValueDataBasedOnQualType(Value &V, unsigned long long Data) {
  QualType QT = V.getType();
  if (const auto *ET = QT->getAs<EnumType>())
    QT = ET->getDecl()->getIntegerType();

  switch (QT->castAs<BuiltinType>()->getKind()) {
  default:
    llvm_unreachable("unknown type kind!");
#define X(type, name)                                                          \
  case BuiltinType::name:                                                      \
    V.set##name(Data);                                                         \
    break;
    REPL_BUILTIN_TYPES
#undef X
  }
}

REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType,
                                    unsigned long long Val) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  SetValueDataBasedOnQualType(VRef, Val);
}

REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType,
                                    float Val) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  VRef.setFloat(Val);
}

REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType,
                                    double Val) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  VRef.setDouble(Val);
}

REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType,
                                    long double Val) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  VRef.setLongDouble(Val);
}
