//===-- cc1_main.cpp - Clang CC1 Compiler Frontend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include <cstdio>

#ifdef CLANG_HAVE_RLIMITS
#include <sys/resource.h>
#endif

using namespace clang;
using namespace llvm::opt;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const char *Message,
                             bool GenCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine *>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  llvm::sys::Process::Exit(GenCrashDiag ? 70 : 1);
}

#ifdef CLANG_HAVE_RLIMITS
/// Attempt to ensure that we have at least 8MiB of usable stack space.
static void ensureSufficientStack() {
  struct rlimit rlim;
  if (getrlimit(RLIMIT_STACK, &rlim) != 0)
    return;

  // Increase the soft stack limit to our desired level, if necessary and
  // possible.
  if (rlim.rlim_cur != RLIM_INFINITY &&
      rlim.rlim_cur < rlim_t(DesiredStackSize)) {
    // Try to allocate sufficient stack.
    if (rlim.rlim_max == RLIM_INFINITY ||
        rlim.rlim_max >= rlim_t(DesiredStackSize))
      rlim.rlim_cur = DesiredStackSize;
    else if (rlim.rlim_cur == rlim.rlim_max)
      return;
    else
      rlim.rlim_cur = rlim.rlim_max;

    if (setrlimit(RLIMIT_STACK, &rlim) != 0 ||
        rlim.rlim_cur != DesiredStackSize)
      return;
  }
}
#else
static void ensureSufficientStack() {}
#endif

/// Print supported cpus of the given target.
static int PrintSupportedCPUs(std::string TargetStr) {
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TargetStr, Error);
  if (!TheTarget) {
    llvm::errs() << Error;
    return 1;
  }

  // the target machine will handle the mcpu printing
  llvm::TargetOptions Options;
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(
      TheTarget->createTargetMachine(TargetStr, "", "+cpuhelp", Options,
                                     std::nullopt));
  return 0;
}

static int PrintSupportedExtensions(std::string TargetStr) {
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TargetStr, Error);
  if (!TheTarget) {
    llvm::errs() << Error;
    return 1;
  }

  llvm::TargetOptions Options;
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(
      TheTarget->createTargetMachine(TargetStr, "", "", Options, std::nullopt));
  const llvm::Triple &MachineTriple = TheTargetMachine->getTargetTriple();
  const llvm::MCSubtargetInfo *MCInfo = TheTargetMachine->getMCSubtargetInfo();
  const llvm::ArrayRef<llvm::SubtargetFeatureKV> Features =
      MCInfo->getAllProcessorFeatures();

  llvm::StringMap<llvm::StringRef> DescMap;
  for (const llvm::SubtargetFeatureKV &feature : Features)
    DescMap.insert({feature.Key, feature.Desc});

  if (MachineTriple.isRISCV())
    llvm::RISCVISAInfo::printSupportedExtensions(DescMap);
  else if (MachineTriple.isAArch64())
    llvm::AArch64::PrintSupportedExtensions();
  else if (MachineTriple.isARM())
    llvm::ARM::PrintSupportedExtensions(DescMap);
  else {
    // The option was already checked in Driver::HandleImmediateArgs,
    // so we do not expect to get here if we are not a supported architecture.
    assert(0 && "Unhandled triple for --print-supported-extensions option.");
    return 1;
  }

  return 0;
}

static int PrintEnabledExtensions(const TargetOptions &TargetOpts) {
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TargetOpts.Triple, Error);
  if (!TheTarget) {
    llvm::errs() << Error;
    return 1;
  }

  // Create a target machine using the input features, the triple information
  // and a dummy instance of llvm::TargetOptions. Note that this is _not_ the
  // same as the `clang::TargetOptions` instance we have access to here.
  llvm::TargetOptions BackendOptions;
  std::string FeaturesStr = llvm::join(TargetOpts.FeaturesAsWritten, ",");
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(
      TheTarget->createTargetMachine(TargetOpts.Triple, TargetOpts.CPU,
                                     FeaturesStr, BackendOptions,
                                     std::nullopt));
  const llvm::Triple &MachineTriple = TheTargetMachine->getTargetTriple();
  const llvm::MCSubtargetInfo *MCInfo = TheTargetMachine->getMCSubtargetInfo();

  // Extract the feature names that are enabled for the given target.
  // We do that by capturing the key from the set of SubtargetFeatureKV entries
  // provided by MCSubtargetInfo, which match the '-target-feature' values.
  const std::vector<llvm::SubtargetFeatureKV> Features =
      MCInfo->getEnabledProcessorFeatures();
  std::set<llvm::StringRef> EnabledFeatureNames;
  for (const llvm::SubtargetFeatureKV &feature : Features)
    EnabledFeatureNames.insert(feature.Key);

  if (MachineTriple.isAArch64())
    llvm::AArch64::printEnabledExtensions(EnabledFeatureNames);
  else if (MachineTriple.isRISCV()) {
    llvm::StringMap<llvm::StringRef> DescMap;
    for (const llvm::SubtargetFeatureKV &feature : Features)
      DescMap.insert({feature.Key, feature.Desc});
    llvm::RISCVISAInfo::printEnabledExtensions(MachineTriple.isArch64Bit(),
                                               EnabledFeatureNames, DescMap);
  } else {
    // The option was already checked in Driver::HandleImmediateArgs,
    // so we do not expect to get here if we are not a supported architecture.
    assert(0 && "Unhandled triple for --print-enabled-extensions option.");
    return 1;
  }

  return 0;
}

int cc1_main(ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr) {
  ensureSufficientStack();

  // Cratels: 编译器实例，类似 clang 编译器的上下文
  // clang-format off
  // Cratels: 单例模式，此处创建后后续就可以直接使用 getCompilerInstance 来获得对象
  // 编译器实例最终会拥有一次编译过程的所有信息，包括源代码路径，用户输入的options等所有信息
  // clang-format on
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  // clang-format off
  // Cratels: PCH是需要额外维护数据的。
  // clang-format on
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  // clang-format off
  // Cratels: 从参数中获取diagnostic的控制参数然后缓存下来交给诊断信息的引擎
  // clang-format on
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (find(Argv, StringRef("-Rround-trip-cc1-args")) != Argv.end())
    Diags.setSeverity(diag::remark_cc1_round_trip_generated,
                      diag::Severity::Remark, {});

  // Cratels: 解析参数，将参数解析之后塞进 Clang
  // clang-format off
  // Cratels: Invocation就是 option
  // 实例中，可见后续Clang->getFrontendOpts()，Clang->getHeaderSearchOpts()等操作
  // 这里CreateFromArgs实际返回值是bool是用来显示创建过程是否正确，实际代码的side effect是改变了Clang->getInvocation()
  // 经过该方法调用后用户调用编译时的option等信息就写入到了Clang对象中，比如源码路径，include路径，options（包括Action种类）等信息。
  // Clang是掌管整个流程，Invocation是一次调用的抽象类
  // 这一步就是为了解析所有参数，后面的步骤中直接使用即可。
  // clang-format on
  bool Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                                    Argv, Diags, Argv0);

  // Cratels: 解析完参数后直接使用
  // clang-format off
  // Cratels: 是否跟踪执行时间
  // clang-format on
  if (!Clang->getFrontendOpts().TimeTracePath.empty()) {
    llvm::timeTraceProfilerInitialize(
        Clang->getFrontendOpts().TimeTraceGranularity, Argv0);
  }

  // clang-format off
  // Cratels: 打印出目标支持的CPU的triple
  // clang-format on
  // --print-supported-cpus takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedCPUs) {
    llvm::outs() << Clang->getTargetOpts().Triple;
    return PrintSupportedCPUs(Clang->getTargetOpts().Triple);
  }

  // --print-supported-extensions takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedExtensions)
    return PrintSupportedExtensions(Clang->getTargetOpts().Triple);

  // --print-enabled-extensions takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintEnabledExtensions)
    return PrintEnabledExtensions(Clang->getTargetOpts());

  // clang-format off
  // Cratels: 如果编译时指定了builtin include的路径，就是用该路径。如果没有指定，就是用默认的路径。
  // clang-format on
  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  // clang-format off
  // Cratels: 创建诊断引擎
  // clang-format on
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  // clang-format off
  // Cratels: 设置错误处理程序，以便任何 LLVM 后端诊断都会通过我们的错误处理程序。
  // clang-format on
  llvm::install_fatal_error_handler(
      LLVMErrorHandler, static_cast<void *>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success) {
    Clang->getDiagnosticClient().finish();
    return 1;
  }

  // Execute the frontend actions.
  // Cratels: 开始执行真正的前端动作
  {
    llvm::TimeTraceScope TimeScope("ExecuteCompiler");
    llvm::outs() << "开始执行真正的前端解析动作......\n";
    // clang-format off
    // Cratels: 上面的步骤初始化了Clang对象持有的CompilerInvocation对象，并且为其准备了一次编译的所有信息，在这里就是执行这个CompilerInvocation，即执行这一次编译。
    // clang-format on
    Success = ExecuteCompilerInvocation(Clang.get());
  }

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    // It is possible that the compiler instance doesn't own a file manager here
    // if we're compiling a module unit. Since the file manager are owned by AST
    // when we're compiling a module unit. So the file manager may be invalid
    // here.
    //
    // It should be fine to create file manager here since the file system
    // options are stored in the compiler invocation and we can recreate the VFS
    // from the compiler invocation.
    if (!Clang->hasFileManager())
      Clang->createFileManager(createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics()));

    if (auto profilerOutput = Clang->createOutputFile(
            Clang->getFrontendOpts().TimeTracePath, /*Binary=*/false,
            /*RemoveFileOnSignal=*/false,
            /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profilerOutput);
      profilerOutput.reset();
      llvm::timeTraceProfilerCleanup();
      Clang->clearOutputFiles(false);
    }
  }

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(Clang));
    return !Success;
  }

  return !Success;
}
