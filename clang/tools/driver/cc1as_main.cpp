//===-- cc1as_main.cpp - Clang Assembler  ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1as functionality, which implements
// the direct interface to the LLVM MC based assembler.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/CC1AsOptions.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/DataLayout.h"
using namespace clang;
using namespace clang::driver;
using namespace llvm;

namespace {

/// \brief Helper class for representing a single invocation of the assembler.
struct AssemblerInvocation {
  /// @name Target Options
  /// @{

  /// The name of the target triple to assemble for.
  std::string Triple;

  /// If given, the name of the target CPU to determine which instructions
  /// are legal.
  std::string CPU;

  /// The list of target specific features to enable or disable -- this should
  /// be a list of strings starting with '+' or '-'.
  std::vector<std::string> Features;

  /// @}
  /// @name Language Options
  /// @{

  std::vector<std::string> IncludePaths;
  unsigned NoInitialTextSection : 1;
  unsigned SaveTemporaryLabels : 1;
  unsigned GenDwarfForAssembly : 1;
  std::string DwarfDebugFlags;

  /// @}
  /// @name Frontend Options
  /// @{

  std::string InputFile;
  std::vector<std::string> LLVMArgs;
  std::string OutputPath;
  enum FileType {
    FT_Asm,  ///< Assembly (.s) output, transliterate mode.
    FT_Null, ///< No output, for timing purposes.
    FT_Obj   ///< Object file output.
  };
  FileType OutputType;
  unsigned ShowHelp : 1;
  unsigned ShowVersion : 1;

  /// @}
  /// @name Transliterate Options
  /// @{

  unsigned OutputAsmVariant;
  unsigned ShowEncoding : 1;
  unsigned ShowInst : 1;

  /// @}
  /// @name Assembler Options
  /// @{

  unsigned RelaxAll : 1;
  unsigned NoExecStack : 1;

  /// @}

public:
  AssemblerInvocation() {
    Triple = "";
    NoInitialTextSection = 0;
    InputFile = "-";
    OutputPath = "-";
    OutputType = FT_Asm;
    OutputAsmVariant = 0;
    ShowInst = 0;
    ShowEncoding = 0;
    RelaxAll = 0;
    NoExecStack = 0;
  }

  static bool CreateFromArgs(AssemblerInvocation &Res, const char **ArgBegin,
                             const char **ArgEnd, DiagnosticsEngine &Diags);
};

}

bool AssemblerInvocation::CreateFromArgs(AssemblerInvocation &Opts,
                                         const char **ArgBegin,
                                         const char **ArgEnd,
                                         DiagnosticsEngine &Diags) {
  using namespace clang::driver::cc1asoptions;
  bool Success = true;

  // Parse the arguments.
  OwningPtr<OptTable> OptTbl(createCC1AsOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  OwningPtr<InputArgList> Args(
    OptTbl->ParseArgs(ArgBegin, ArgEnd,MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (arg_iterator it = Args->filtered_begin(cc1asoptions::OPT_UNKNOWN),
         ie = Args->filtered_end(); it != ie; ++it) {
    Diags.Report(diag::err_drv_unknown_argument) << (*it) ->getAsString(*Args);
    Success = false;
  }

  // Construct the invocation.

  // Target Options
  Opts.Triple = llvm::Triple::normalize(Args->getLastArgValue(OPT_triple));
  Opts.CPU = Args->getLastArgValue(OPT_target_cpu);
  Opts.Features = Args->getAllArgValues(OPT_target_feature);

  // Use the default target triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getDefaultTargetTriple();

  // Language Options
  Opts.IncludePaths = Args->getAllArgValues(OPT_I);
  Opts.NoInitialTextSection = Args->hasArg(OPT_n);
  Opts.SaveTemporaryLabels = Args->hasArg(OPT_L);
  Opts.GenDwarfForAssembly = Args->hasArg(OPT_g);
  Opts.DwarfDebugFlags = Args->getLastArgValue(OPT_dwarf_debug_flags);

  // Frontend Options
  if (Args->hasArg(OPT_INPUT)) {
    bool First = true;
    for (arg_iterator it = Args->filtered_begin(OPT_INPUT),
           ie = Args->filtered_end(); it != ie; ++it, First=false) {
      const Arg *A = it;
      if (First)
        Opts.InputFile = A->getValue();
      else {
        Diags.Report(diag::err_drv_unknown_argument) << A->getAsString(*Args);
        Success = false;
      }
    }
  }
  Opts.LLVMArgs = Args->getAllArgValues(OPT_mllvm);
  if (Args->hasArg(OPT_fatal_warnings))
    Opts.LLVMArgs.push_back("-fatal-assembler-warnings");
  Opts.OutputPath = Args->getLastArgValue(OPT_o);
  if (Arg *A = Args->getLastArg(OPT_filetype)) {
    StringRef Name = A->getValue();
    unsigned OutputType = StringSwitch<unsigned>(Name)
      .Case("asm", FT_Asm)
      .Case("null", FT_Null)
      .Case("obj", FT_Obj)
      .Default(~0U);
    if (OutputType == ~0U) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(*Args) << Name;
      Success = false;
    } else
      Opts.OutputType = FileType(OutputType);
  }
  Opts.ShowHelp = Args->hasArg(OPT_help);
  Opts.ShowVersion = Args->hasArg(OPT_version);

  // Transliterate Options
  Opts.OutputAsmVariant = Args->getLastArgIntValue(OPT_output_asm_variant,
                                                   0, Diags);
  Opts.ShowEncoding = Args->hasArg(OPT_show_encoding);
  Opts.ShowInst = Args->hasArg(OPT_show_inst);

  // Assemble Options
  Opts.RelaxAll = Args->hasArg(OPT_relax_all);
  Opts.NoExecStack =  Args->hasArg(OPT_no_exec_stack);

  return Success;
}

static formatted_raw_ostream *GetOutputStream(AssemblerInvocation &Opts,
                                              DiagnosticsEngine &Diags,
                                              bool Binary) {
  if (Opts.OutputPath.empty())
    Opts.OutputPath = "-";

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (Opts.OutputPath != "-")
    sys::RemoveFileOnSignal(sys::Path(Opts.OutputPath));

  std::string Error;
  raw_fd_ostream *Out =
    new raw_fd_ostream(Opts.OutputPath.c_str(), Error,
                       (Binary ? raw_fd_ostream::F_Binary : 0));
  if (!Error.empty()) {
    Diags.Report(diag::err_fe_unable_to_open_output)
      << Opts.OutputPath << Error;
    return 0;
  }

  return new formatted_raw_ostream(*Out, formatted_raw_ostream::DELETE_STREAM);
}

static bool ExecuteAssembler(AssemblerInvocation &Opts,
                             DiagnosticsEngine &Diags) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget(TargetRegistry::lookupTarget(Opts.Triple, Error));
  if (!TheTarget) {
    Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
    return false;
  }

  OwningPtr<MemoryBuffer> BufferPtr;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Opts.InputFile, BufferPtr)) {
    Error = ec.message();
    Diags.Report(diag::err_fe_error_reading) << Opts.InputFile;
    return false;
  }
  MemoryBuffer *Buffer = BufferPtr.take();

  SourceMgr SrcMgr;

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(Opts.IncludePaths);

  OwningPtr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(Opts.Triple));
  assert(MAI && "Unable to create target asm info!");

  OwningPtr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(Opts.Triple));
  assert(MRI && "Unable to create target register info!");

  bool IsBinary = Opts.OutputType == AssemblerInvocation::FT_Obj;
  formatted_raw_ostream *Out = GetOutputStream(Opts, Diags, IsBinary);
  if (!Out)
    return false;

  // FIXME: This is not pretty. MCContext has a ptr to MCObjectFileInfo and
  // MCObjectFileInfo needs a MCContext reference in order to initialize itself.
  OwningPtr<MCObjectFileInfo> MOFI(new MCObjectFileInfo());
  MCContext Ctx(*MAI, *MRI, MOFI.get(), &SrcMgr);
  // FIXME: Assembler behavior can change with -static.
  MOFI->InitMCObjectFileInfo(Opts.Triple,
                             Reloc::Default, CodeModel::Default, Ctx);
  if (Opts.SaveTemporaryLabels)
    Ctx.setAllowTemporaryLabels(false);
  if (Opts.GenDwarfForAssembly)
    Ctx.setGenDwarfForAssembly(true);
  if (!Opts.DwarfDebugFlags.empty())
    Ctx.setDwarfDebugFlags(StringRef(Opts.DwarfDebugFlags));

  // Build up the feature string from the target feature list.
  std::string FS;
  if (!Opts.Features.empty()) {
    FS = Opts.Features[0];
    for (unsigned i = 1, e = Opts.Features.size(); i != e; ++i)
      FS += "," + Opts.Features[i];
  }

  OwningPtr<MCStreamer> Str;

  OwningPtr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  OwningPtr<MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(Opts.Triple, Opts.CPU, FS));

  // FIXME: There is a bit of code duplication with addPassesToEmitFile.
  if (Opts.OutputType == AssemblerInvocation::FT_Asm) {
    MCInstPrinter *IP =
      TheTarget->createMCInstPrinter(Opts.OutputAsmVariant, *MAI, *MCII, *MRI,
                                     *STI);
    MCCodeEmitter *CE = 0;
    MCAsmBackend *MAB = 0;
    if (Opts.ShowEncoding) {
      CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, *STI, Ctx);
      MAB = TheTarget->createMCAsmBackend(Opts.Triple, Opts.CPU);
    }
    Str.reset(TheTarget->createAsmStreamer(Ctx, *Out, /*asmverbose*/true,
                                           /*useLoc*/ true,
                                           /*useCFI*/ true,
                                           /*useDwarfDirectory*/ true,
                                           IP, CE, MAB,
                                           Opts.ShowInst));
  } else if (Opts.OutputType == AssemblerInvocation::FT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(Opts.OutputType == AssemblerInvocation::FT_Obj &&
           "Invalid file type!");
    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, *STI, Ctx);
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(Opts.Triple, Opts.CPU);
    Str.reset(TheTarget->createMCObjectStreamer(Opts.Triple, Ctx, *MAB, *Out,
                                                CE, Opts.RelaxAll,
                                                Opts.NoExecStack));
    Str.get()->InitSections();
  }

  OwningPtr<MCAsmParser> Parser(createMCAsmParser(SrcMgr, Ctx,
                                                  *Str.get(), *MAI));
  OwningPtr<MCTargetAsmParser> TAP(TheTarget->createMCAsmParser(*STI, *Parser));
  if (!TAP) {
    Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
    return false;
  }

  Parser->setTargetParser(*TAP.get());

  bool Success = !Parser->Run(Opts.NoInitialTextSection);

  // Close the output.
  delete Out;

  // Delete output on errors.
  if (!Success && Opts.OutputPath != "-")
    sys::Path(Opts.OutputPath).eraseFromDisk();

  return Success;
}

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

int cc1as_main(const char **ArgBegin, const char **ArgEnd,
               const char *Argv0, void *MainAddr) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(ArgEnd - ArgBegin, ArgBegin);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();

  // Construct our diagnostic client.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient
    = new TextDiagnosticPrinter(errs(), &*DiagOpts);
  DiagClient->setPrefix("clang -cc1as");
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  ScopedFatalErrorHandler FatalErrorHandler
    (LLVMErrorHandler, static_cast<void*>(&Diags));

  // Parse the arguments.
  AssemblerInvocation Asm;
  if (!AssemblerInvocation::CreateFromArgs(Asm, ArgBegin, ArgEnd, Diags))
    return 1;

  // Honor -help.
  if (Asm.ShowHelp) {
    OwningPtr<driver::OptTable> Opts(driver::createCC1AsOptTable());
    Opts->PrintHelp(llvm::outs(), "clang -cc1as", "Clang Integrated Assembler");
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Asm.ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  if (!Asm.LLVMArgs.empty()) {
    unsigned NumArgs = Asm.LLVMArgs.size();
    const char **Args = new const char*[NumArgs + 2];
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Asm.LLVMArgs[i].c_str();
    Args[NumArgs + 1] = 0;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args);
  }

  // Execute the invocation, unless there were parsing errors.
  bool Success = false;
  if (!Diags.hasErrorOccurred())
    Success = ExecuteAssembler(Asm, Diags);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.
  TimerGroup::printAll(errs());

  return !Success;
}
