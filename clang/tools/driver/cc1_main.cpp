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
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
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
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
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

// clang-format off
// Cratels:作为前端使用
// clang-format on
int cc1_main(ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr) {
  ensureSufficientStack();

  // clang-format off
  // Cratels:Clang 对象是编译器的实例，拥有编译器进行一次编译的所有信息。这里其实采用了单例模式
  // clang-format on
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Initialize targets first, so that --version shows registered targets.
  // clang-format off
  // Cratels:再来一次？？
  // clang-format on
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (find(Argv, StringRef("-Rround-trip-cc1-args")) != Argv.end())
    Diags.setSeverity(diag::remark_cc1_round_trip_generated,
                      diag::Severity::Remark, {});

  // clang-format off
  // Cratels:CreateFromArgs 其实是从 Args 命令行参数中解析所有数据并将其写回到 Clang的 Invocation 属性中
  // Cratels:Invocation 抽象了一个编译器的编译过程
  // clang-format on
  bool Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                                    Argv, Diags, Argv0);

  // clang-format off
  // Cratels:根据解析出来的 option 来进行对应的处理，比如是否计时，是否答应出支持的 CPU 的信息等等
  // clang-format on
  if (!Clang->getFrontendOpts().TimeTracePath.empty()) {
    llvm::timeTraceProfilerInitialize(
        Clang->getFrontendOpts().TimeTraceGranularity, Argv0,
        Clang->getFrontendOpts().TimeTraceVerbose);
  }
  // --print-supported-cpus takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedCPUs)
    return PrintSupportedCPUs(Clang->getTargetOpts().Triple);

  // --print-supported-extensions takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedExtensions)
    return PrintSupportedExtensions(Clang->getTargetOpts().Triple);

  // --print-enabled-extensions takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintEnabledExtensions)
    return PrintEnabledExtensions(Clang->getTargetOpts());

  // clang-format off
  // Cratels:推导出系统默认的 include 路径
  // clang-format on
  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(
      LLVMErrorHandler, static_cast<void *>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success) {
    Clang->getDiagnosticClient().finish();
    return 1;
  }

  // Execute the frontend actions.
  {
    llvm::TimeTraceScope TimeScope("ExecuteCompiler");

    // clang-format off
    // Cratels:执行编译器（Compilance）的一次编译（Invocation）
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
  // clang-format off
  // Cratels:可以控制一次编译器编译之后 Clang 对象不销毁
  // clang-format on
  if (Clang->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(Clang));
    return !Success;
  }

  return !Success;
}
