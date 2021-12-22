/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-compiler.h"
#include "comgr-device-libs.h"
#include "comgr-env.h"
#include "lld/Common/Driver.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

#include "time-stat/ts-interface.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace TimeStatistics;

namespace COMGR {

namespace {
static constexpr llvm::StringLiteral LinkerJobName = "amdgpu::Linker";

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

  /// The list of symbol definitions.
  std::vector<std::string> SymbolDefs;

  /// @}
  /// @name Language Options
  /// @{

  std::vector<std::string> IncludePaths;
  unsigned NoInitialTextSection : 1;
  unsigned SaveTemporaryLabels : 1;
  unsigned GenDwarfForAssembly : 1;
  unsigned RelaxELFRelocations : 1;
  unsigned DwarfVersion;
  std::string DwarfDebugFlags;
  std::string DwarfDebugProducer;
  std::string DebugCompilationDir;
  llvm::DebugCompressionType CompressDebugSections =
      llvm::DebugCompressionType::None;
  std::string MainFileName;

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
  unsigned FatalWarnings : 1;
  unsigned IncrementalLinkerCompatible : 1;

  /// The name of the relocation model to use.
  std::string RelocationModel;

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
    FatalWarnings = 0;
    IncrementalLinkerCompatible = 0;
    DwarfVersion = 0;
  }

  static bool createFromArgs(AssemblerInvocation &Res,
                             ArrayRef<const char *> Argv,
                             DiagnosticsEngine &Diags);
};
} // namespace

bool AssemblerInvocation::createFromArgs(AssemblerInvocation &Opts,
                                         ArrayRef<const char *> Argv,
                                         DiagnosticsEngine &Diags) {
  bool Success = true;

  // Parse the arguments.
  const OptTable &OptTbl = getDriverOptTable();

  const unsigned IncludedFlagsBitmask = options::CC1AsOption;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = OptTbl.ParseArgs(Argv, MissingArgIndex, MissingArgCount,
                                       IncludedFlagsBitmask);

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (const Arg *A : Args.filtered(OPT_UNKNOWN)) {
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (OptTbl.findNearest(ArgString, Nearest, IncludedFlagsBitmask) > 1) {
      Diags.Report(diag::err_drv_unknown_argument) << ArgString;
    } else {
      Diags.Report(diag::err_drv_unknown_argument_with_suggestion)
          << ArgString << Nearest;
    }
    Success = false;
  }

  // Construct the invocation.

  // Target Options
  Opts.Triple = llvm::Triple::normalize(Args.getLastArgValue(OPT_triple));
  Opts.CPU = std::string(Args.getLastArgValue(OPT_target_cpu));
  Opts.Features = Args.getAllArgValues(OPT_target_feature);

  // Use the default target triple if unspecified.
  if (Opts.Triple.empty()) {
    Opts.Triple = llvm::sys::getDefaultTargetTriple();
  }

  // Language Options
  Opts.IncludePaths = Args.getAllArgValues(OPT_I);
  Opts.NoInitialTextSection = Args.hasArg(OPT_n);
  Opts.SaveTemporaryLabels = Args.hasArg(OPT_msave_temp_labels);
  // Any DebugInfoKind implies GenDwarfForAssembly.
  Opts.GenDwarfForAssembly = Args.hasArg(OPT_debug_info_kind_EQ);

  if (const Arg *A = Args.getLastArg(OPT_compress_debug_sections,
                                     OPT_compress_debug_sections_EQ)) {
    if (A->getOption().getID() == OPT_compress_debug_sections) {
      // TODO: be more clever about the compression type auto-detection
      Opts.CompressDebugSections = llvm::DebugCompressionType::GNU;
    } else {
      Opts.CompressDebugSections =
          llvm::StringSwitch<llvm::DebugCompressionType>(A->getValue())
              .Case("none", llvm::DebugCompressionType::None)
              .Case("zlib", llvm::DebugCompressionType::Z)
              .Case("zlib-gnu", llvm::DebugCompressionType::GNU)
              .Default(llvm::DebugCompressionType::None);
    }
  }

  Opts.RelaxELFRelocations = Args.hasArg(OPT_mrelax_relocations);
  Opts.DwarfVersion = getLastArgIntValue(Args, OPT_dwarf_version_EQ, 2, Diags);
  Opts.DwarfDebugFlags =
      std::string(Args.getLastArgValue(OPT_dwarf_debug_flags));
  Opts.DwarfDebugProducer =
      std::string(Args.getLastArgValue(OPT_dwarf_debug_producer));
  Opts.DebugCompilationDir =
      std::string(Args.getLastArgValue(OPT_fdebug_compilation_dir));
  Opts.MainFileName = std::string(Args.getLastArgValue(OPT_main_file_name));

  // Frontend Options
  if (Args.hasArg(OPT_INPUT)) {
    bool First = true;
    for (const Arg *A : Args.filtered(OPT_INPUT)) {
      if (First) {
        Opts.InputFile = A->getValue();
        First = false;
      } else {
        Diags.Report(diag::err_drv_unknown_argument) << A->getAsString(Args);
        Success = false;
      }
    }
  }
  Opts.LLVMArgs = Args.getAllArgValues(OPT_mllvm);
  Opts.OutputPath = std::string(Args.getLastArgValue(OPT_o));
  if (Arg *A = Args.getLastArg(OPT_filetype)) {
    StringRef Name = A->getValue();
    unsigned OutputType = StringSwitch<unsigned>(Name)
                              .Case("asm", FT_Asm)
                              .Case("null", FT_Null)
                              .Case("obj", FT_Obj)
                              .Default(~0U);
    if (OutputType == ~0U) {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.OutputType = FileType(OutputType);
    }
  }
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowVersion = Args.hasArg(OPT_version);

  // Transliterate Options
  Opts.OutputAsmVariant =
      getLastArgIntValue(Args, OPT_output_asm_variant, 0, Diags);
  Opts.ShowEncoding = Args.hasArg(OPT_show_encoding);
  Opts.ShowInst = Args.hasArg(OPT_show_inst);

  // Assemble Options
  Opts.RelaxAll = Args.hasArg(OPT_mrelax_all);
  Opts.NoExecStack = Args.hasArg(OPT_mno_exec_stack);
  Opts.FatalWarnings = Args.hasArg(OPT_massembler_fatal_warnings);
  Opts.RelocationModel =
      std::string(Args.getLastArgValue(OPT_mrelocation_model, "pic"));
  Opts.IncrementalLinkerCompatible =
      Args.hasArg(OPT_mincremental_linker_compatible);
  Opts.SymbolDefs = Args.getAllArgValues(OPT_defsym);

  return Success;
}

static std::unique_ptr<raw_fd_ostream>
getOutputStream(AssemblerInvocation &Opts, DiagnosticsEngine &Diags,
                bool Binary) {
  if (Opts.OutputPath.empty()) {
    Opts.OutputPath = "-";
  }

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (Opts.OutputPath != "-") {
    sys::RemoveFileOnSignal(Opts.OutputPath);
  }

  std::error_code EC;
  auto Out = std::make_unique<raw_fd_ostream>(
      Opts.OutputPath, EC, (Binary ? sys::fs::OF_None : sys::fs::OF_Text));
  if (EC) {
    Diags.Report(diag::err_fe_unable_to_open_output)
        << Opts.OutputPath << EC.message();
    return nullptr;
  }

  return Out;
}

static bool executeAssemblerImpl(AssemblerInvocation &Opts,
                                 DiagnosticsEngine &Diags, raw_ostream &LogS) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(Opts.Triple, Error);
  if (!TheTarget) {
    return Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFileOrSTDIN(Opts.InputFile);

  if (std::error_code EC = Buffer.getError()) {
    Error = EC.message();
    return Diags.Report(diag::err_fe_error_reading) << Opts.InputFile;
  }

  SourceMgr SrcMgr;
  SrcMgr.setDiagHandler(
      [](const SMDiagnostic &SMDiag, void *LogS) {
        SMDiag.print("", *(raw_ostream *)LogS, /* ShowColors */ false);
      },
      &LogS);

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(*Buffer), SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(Opts.IncludePaths);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(Opts.Triple));
  assert(MRI && "Unable to create target register info!");

  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, Opts.Triple, MCOptions));
  assert(MAI && "Unable to create target asm info!");

  // Ensure MCAsmInfo initialization occurs before any use, otherwise sections
  // may be created with a combination of default and explicit settings.
  MAI->setCompressDebugSections(Opts.CompressDebugSections);

  MAI->setRelaxELFRelocations(Opts.RelaxELFRelocations);

  bool IsBinary = Opts.OutputType == AssemblerInvocation::FT_Obj;
  std::unique_ptr<raw_fd_ostream> FDOS = getOutputStream(Opts, Diags, IsBinary);
  if (!FDOS) {
    return true;
  }

  // Build up the feature string from the target feature list.
  std::string FS;
  if (!Opts.Features.empty()) {
    FS = Opts.Features[0];
    for (unsigned I = 1, E = Opts.Features.size(); I != E; ++I) {
      FS += "," + Opts.Features[I];
    }
  }

  std::unique_ptr<MCObjectFileInfo> MOFI(new MCObjectFileInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(Opts.Triple, Opts.CPU, FS));

  MCContext Ctx(Triple(Opts.Triple), MAI.get(), MRI.get(),
                STI.get(), &SrcMgr);
  Ctx.setObjectFileInfo(MOFI.get());

  bool PIC = false;
  if (Opts.RelocationModel == "static") {
    PIC = false;
  } else if (Opts.RelocationModel == "pic") {
    PIC = true;
  } else {
    assert(Opts.RelocationModel == "dynamic-no-pic" && "Invalid PIC model!");
    PIC = false;
  }

  MOFI->initMCObjectFileInfo(Ctx, PIC);
  if (Opts.SaveTemporaryLabels) {
    Ctx.setAllowTemporaryLabels(false);
  }
  if (Opts.GenDwarfForAssembly) {
    Ctx.setGenDwarfForAssembly(true);
  }
  if (!Opts.DwarfDebugFlags.empty()) {
    Ctx.setDwarfDebugFlags(StringRef(Opts.DwarfDebugFlags));
  }
  if (!Opts.DwarfDebugProducer.empty()) {
    Ctx.setDwarfDebugProducer(StringRef(Opts.DwarfDebugProducer));
  }
  if (!Opts.DebugCompilationDir.empty()) {
    Ctx.setCompilationDir(Opts.DebugCompilationDir);
  }
  if (!Opts.MainFileName.empty()) {
    Ctx.setMainFileName(StringRef(Opts.MainFileName));
  }
  Ctx.setDwarfVersion(Opts.DwarfVersion);

  std::unique_ptr<MCStreamer> Str;
  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());

  raw_pwrite_stream *Out = FDOS.get();
  std::unique_ptr<buffer_ostream> BOS;

  // FIXME: There is a bit of code duplication with addPassesToEmitFile.
  if (Opts.OutputType == AssemblerInvocation::FT_Asm) {
    MCInstPrinter *IP = TheTarget->createMCInstPrinter(
        llvm::Triple(Opts.Triple), Opts.OutputAsmVariant, *MAI, *MCII, *MRI);
    std::unique_ptr<MCCodeEmitter> MCE;
    std::unique_ptr<MCAsmBackend> MAB;
    if (Opts.ShowEncoding) {
      MCE.reset(TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx));
      MCTargetOptions Options;
      MAB.reset(TheTarget->createMCAsmBackend(*STI, *MRI, Options));
    }
    auto FOut = std::make_unique<formatted_raw_ostream>(*Out);
    Str.reset(TheTarget->createAsmStreamer(
        Ctx, std::move(FOut), /*asmverbose*/ true,
        /*useDwarfDirectory*/ true, IP, std::move(MCE), std::move(MAB),
        Opts.ShowInst));
  } else if (Opts.OutputType == AssemblerInvocation::FT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(Opts.OutputType == AssemblerInvocation::FT_Obj &&
           "Invalid file type!");
    if (!FDOS->supportsSeeking()) {
      BOS = std::make_unique<buffer_ostream>(*FDOS);
      Out = BOS.get();
    }

    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
    MCTargetOptions Options;
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, Options);
    Triple T(Opts.Triple);
    Str.reset(TheTarget->createMCObjectStreamer(
        T, Ctx, std::unique_ptr<MCAsmBackend>(MAB),
        MAB->createObjectWriter(*Out), std::unique_ptr<MCCodeEmitter>(CE), *STI,
        Opts.RelaxAll, Opts.IncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ true));
    Str.get()->initSections(Opts.NoExecStack, *STI);
  }

  bool Failed = false;

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *Str.get(), *MAI));

  // FIXME: init MCTargetOptions from sanitizer flags here.
  MCTargetOptions Options;
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, Options));
  if (!TAP) {
    Failed = Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
  }

  // Set values for symbols, if any.
  for (auto &S : Opts.SymbolDefs) {
    auto Pair = StringRef(S).split('=');
    auto Sym = Pair.first;
    auto Val = Pair.second;
    int64_t Value;
    // We have already error checked this in the driver.
    if (!Val.getAsInteger(0, Value)) {
      Ctx.setSymbolValue(Parser->getStreamer(), Sym, Value);
    }
  }

  if (!Failed) {
    Parser->setTargetParser(*TAP.get());
    Failed = Parser->Run(Opts.NoInitialTextSection);
  }

  return Failed;
}

static bool executeAssembler(AssemblerInvocation &Opts,
                             DiagnosticsEngine &Diags, raw_ostream &LogS) {
  bool Failed = executeAssemblerImpl(Opts, Diags, LogS);

  // Delete output file if there were errors.
  if (Failed && Opts.OutputPath != "-") {
    sys::fs::remove(Opts.OutputPath);
  }

  return Failed;
}

static SmallString<128> getFilePath(DataObject *Object, StringRef Dir) {
  SmallString<128> Path(Dir);
  path::append(Path, Object->Name);
  return Path;
}

static amd_comgr_status_t inputFromFile(DataObject *Object, StringRef Path) {
  ProfilePoint Point("FileIO");
  auto BufOrError = MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufOrError.getError()) {
    return AMD_COMGR_STATUS_ERROR;
  }
  Object->setData(BufOrError.get()->getBuffer());
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t outputToFile(StringRef Data, StringRef Path) {
  SmallString<128> DirPath = Path;
  path::remove_filename(DirPath);
  {
    ProfilePoint Point("CreateDir");
    if (fs::create_directories(DirPath)) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
  std::error_code EC;
  ProfilePoint Point("FileIO");
  raw_fd_ostream OS(Path, EC, fs::OF_None);
  if (EC) {
    return AMD_COMGR_STATUS_ERROR;
  }
  OS << Data;
  OS.close();
  if (OS.has_error()) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t outputToFile(DataObject *Object, StringRef Path) {
  return outputToFile(StringRef(Object->Data, Object->Size), Path);
}

static void initializeCommandLineArgs(SmallVectorImpl<const char *> &Args) {
  // Workaround for flawed Driver::BuildCompilation(...) implementation,
  // which eliminates 1st argument, cause it actually awaits argv[0].
  Args.clear();
  Args.push_back("");
}

// Parse -mllvm options
static amd_comgr_status_t
parseLLVMOptions(const std::vector<std::string> &Options) {
  std::vector<const char *> LLVMArgs;
  for (auto Option : Options) {
    LLVMArgs.push_back("");
    LLVMArgs.push_back(Option.c_str());
    if (!cl::ParseCommandLineOptions(LLVMArgs.size(), &LLVMArgs[0],
                                     "-mllvm options parsing")) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    LLVMArgs.clear();
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t linkWithLLD(llvm::ArrayRef<const char *> Args,
                                      llvm::raw_ostream &LogS,
                                      llvm::raw_ostream &LogE) {
  ArgStringList LLDArgs(llvm::iterator_range<ArrayRef<const char *>::iterator>(
      Args.begin(), Args.end()));
  LLDArgs.insert(LLDArgs.begin(), "lld");
  LLDArgs.push_back("--threads=1");
  ArrayRef<const char *> ArgRefs = llvm::makeArrayRef(LLDArgs);
  static std::mutex MScreen;
  MScreen.lock();
  bool LLDRet = lld::elf::link(ArgRefs, false, LogS, LogE);
  MScreen.unlock();
  if (!LLDRet) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static void logArgv(raw_ostream &OS, StringRef ProgramName,
                    ArrayRef<const char *> Argv) {
  OS << "COMGR::executeInProcessDriver argv: " << ProgramName;
  for (size_t I = 0; I < Argv.size(); ++I) {
    // Skip the first argument, which we replace with ProgramName, and the last
    // argument, which is a null terminator.
    if (I && Argv[I]) {
      OS << " \"" << Argv[I] << '\"';
    }
  }
  OS << '\n';
}

amd_comgr_status_t
AMDGPUCompiler::executeInProcessDriver(ArrayRef<const char *> Args) {
  // A DiagnosticsEngine is required at several points:
  //  * By the Driver in order to diagnose option parsing.
  //  * By the CompilerInvocation in order to diagnose option parsing.
  //  * By the CompilerInstance in order to diagnose everything else.
  // It is a chicken-and-egg problem in that you need some form of diagnostics
  // in order to diagnose options which further influence diagnostics. The code
  // here is mostly copy-and-pasted from driver.cpp/cc1_main.cpp/various Clang
  // tests to try to approximate the same behavior as running the `clang`
  // executable.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions);
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList ArgList = getDriverOptTable().ParseArgs(
      Args.slice(1), MissingArgIndex, MissingArgCount);
  // We ignore MissingArgCount and the return value of ParseDiagnosticArgs. Any
  // errors that would be diagnosed here will also be diagnosed later, when the
  // DiagnosticsEngine actually exists.
  (void)ParseDiagnosticArgs(*DiagOpts, ArgList);
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(LogS, &*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs);
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  ProcessWarningOptions(Diags, *DiagOpts, /*ReportDiags=*/false);
  Driver TheDriver("", "", Diags);
  TheDriver.setTitle("AMDGPU Code Object Manager");
  TheDriver.setCheckInputsExist(false);

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C) {
    return C->containsError() ? AMD_COMGR_STATUS_ERROR
                              : AMD_COMGR_STATUS_SUCCESS;
  }
  for (auto &Job : C->getJobs()) {
    auto Arguments = Job.getArguments();
    SmallVector<const char *, 128> Argv;
    initializeCommandLineArgs(Argv);
    Argv.append(Arguments.begin(), Arguments.end());
    Argv.push_back(nullptr);

    // By default clang driver will ask CC1 to leak memory.
    auto *IT = find(Argv, StringRef("-disable-free"));
    if (IT != Argv.end()) {
      Argv.erase(IT);
    }

    clearLLVMOptions();

    if (Argv[1] == StringRef("-cc1")) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "clang", Argv);
      }
      std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
      Clang->setVerboseOutputStream(LogS);
      if (!Argv.back()) {
        Argv.pop_back();
      }
      if (!CompilerInvocation::CreateFromArgs(Clang->getInvocation(), Argv,
                                              Diags)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      // Internally this call refers to the invocation created above, so at
      // this point the DiagnosticsEngine should accurately reflect all user
      // requested configuration from Argv.
      Clang->createDiagnostics(DiagClient, /* ShouldOwnClient */ false);
      if (!Clang->hasDiagnostics()) {
        return AMD_COMGR_STATUS_ERROR;
      }
      if (!ExecuteCompilerInvocation(Clang.get())) {
        return AMD_COMGR_STATUS_ERROR;
      }
    } else if (Argv[1] == StringRef("-cc1as")) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "clang", Argv);
      }
      Argv.erase(Argv.begin() + 1);
      if (!Argv.back()) {
        Argv.pop_back();
      }
      AssemblerInvocation Asm;
      if (!AssemblerInvocation::createFromArgs(Asm, Argv, Diags)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      if (auto Status = parseLLVMOptions(Asm.LLVMArgs)) {
        return Status;
      }
      if (executeAssembler(Asm, Diags, LogS)) {
        return AMD_COMGR_STATUS_ERROR;
      }
    } else if (Job.getCreator().getName() == LinkerJobName) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "lld", Argv);
      }
      if (auto Status = linkWithLLD(Arguments, LogS, LogS)) {
        return Status;
      }
    } else {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::createTmpDirs() {
  ProfilePoint Point("CreateDir");
  if (fs::createUniqueDirectory("comgr", TmpDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  InputDir = TmpDir;
  path::append(InputDir, "input");
  if (fs::create_directory(InputDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  OutputDir = TmpDir;
  path::append(OutputDir, "output");
  if (fs::create_directory(OutputDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  IncludeDir = TmpDir;
  path::append(IncludeDir, "include");
  if (fs::create_directory(IncludeDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

// On windows fs::remove_directories takes huge time so use fs::remove.
amd_comgr_status_t RemoveDirectory(const StringRef DirName) {
  std::error_code EC;
  for (fs::directory_iterator Dir(DirName, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    const StringRef Path = Dir->path();

    fs::file_status Status;
    EC = fs::status(Path, Status);
    if (EC) {
      return AMD_COMGR_STATUS_ERROR;
    }

    switch (Status.type()) {
    case fs::file_type::regular_file:
      if (fs::remove(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      break;
    case fs::file_type::directory_file:
      if (RemoveDirectory(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }

      if (fs::remove(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      break;
    default:
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  if (fs::remove(DirName)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::removeTmpDirs() {
  if (TmpDir.empty()) {
    return AMD_COMGR_STATUS_SUCCESS;
  }
  ProfilePoint Point("RemoveDir");
#ifndef _WIN32
  if (fs::remove_directories(TmpDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
#else
  return RemoveDirectory(TmpDir);
#endif
}

amd_comgr_status_t AMDGPUCompiler::executeOutOfProcessHIPCompilation(
    llvm::ArrayRef<const char *> Args) {
  std::string Exec = (Twine(env::getHIPPath()) + "/bin/hipcc").str();
  std::vector<StringRef> ArgsV;
  ArgsV.push_back(Exec);
  for (unsigned I = 0, E = Args.size(); I != E; ++I) {
    if (strcmp(Args[I], "-hip-path") == 0) {
      ++I;
      if (I == E) {
        LogS << "Error: -hip-path option misses argument.\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      Exec = (Twine(Args[I]) + "/bin/hipcc").str();
      ArgsV[0] = Exec;

    } else {
      ArgsV.push_back(Args[I]);
    }
  }

  ArgsV.push_back("--genco");
  std::vector<Optional<StringRef>> Redirects;
  std::string ErrMsg;
  int RC = sys::ExecuteAndWait(Exec, ArgsV,
                               /*env=*/None, Redirects, /*secondsToWait=*/0,
                               /*memoryLimit=*/0, &ErrMsg);
  LogS << ErrMsg;
  return RC ? AMD_COMGR_STATUS_ERROR : AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::processFile(const char *InputFilePath,
                                               const char *OutputFilePath) {
  SmallVector<const char *, 128> Argv;

  for (auto &Arg : Args) {
    Argv.push_back(Arg);
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Argv.push_back(Option.c_str());
    if (Option.rfind("--rocm-path", 0) == 0) {
      NoGpuLib = false;
    }
  }

  Argv.push_back(InputFilePath);

  // By default, disable bitcode selection and linking by the driver.
  // FIXME: We should always let the driver take care of bitcode library
  // selection and linking when we have a consistent path to use.
  if (NoGpuLib) {
    Argv.push_back("-nogpulib");
  }

  Argv.push_back("-o");
  Argv.push_back(OutputFilePath);

  // For HIP OOP compilation, we launch a process.
  if (CompileOOP && getLanguage() == AMD_COMGR_LANGUAGE_HIP) {
    return executeOutOfProcessHIPCompilation(Argv);
  }

  return executeInProcessDriver(Argv);
}

amd_comgr_status_t
AMDGPUCompiler::processFiles(amd_comgr_data_kind_t OutputKind,
                             const char *OutputSuffix) {
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_INCLUDE) {
      continue;
    }
    auto IncludeFilePath = getFilePath(Input, IncludeDir);
    if (auto Status = outputToFile(Input, IncludeFilePath)) {
      return Status;
    }
  }

  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_SOURCE &&
        Input->DataKind != AMD_COMGR_DATA_KIND_BC &&
        Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE &&
        Input->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE) {
      continue;
    }

    auto InputFilePath = getFilePath(Input, InputDir);
    if (auto Status = outputToFile(Input, InputFilePath)) {
      return Status;
    }

    amd_comgr_data_t OutputT;
    if (auto Status = amd_comgr_create_data(OutputKind, &OutputT)) {
      return Status;
    }
    ScopedDataObjectReleaser SDOR(OutputT);

    DataObject *Output = DataObject::convert(OutputT);
    Output->setName(std::string(Input->Name) + OutputSuffix);

    auto OutputFilePath = getFilePath(Output, OutputDir);

    if (auto Status =
            processFile(InputFilePath.c_str(), OutputFilePath.c_str())) {
      return Status;
    }

    if (auto Status = inputFromFile(Output, OutputFilePath)) {
      return Status;
    }

    if (auto Status = amd_comgr_data_set_add(OutSetT, OutputT)) {
      return Status;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::addIncludeFlags() {
  if (ActionInfo->Path) {
    Args.push_back("-I");
    Args.push_back(ActionInfo->Path);
  }

  Args.push_back("-I");
  Args.push_back(IncludeDir.c_str());

  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER) {
      continue;
    }
    PrecompiledHeaders.push_back(getFilePath(Input, IncludeDir));
    auto &PrecompiledHeaderPath = PrecompiledHeaders.back();
    if (auto Status = outputToFile(Input, PrecompiledHeaderPath)) {
      return Status;
    }
    Args.push_back("-include-pch");
    Args.push_back(PrecompiledHeaderPath.c_str());
    Args.push_back("-Xclang");
    Args.push_back("-fno-validate-pch");
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
AMDGPUCompiler::addTargetIdentifierFlags(llvm::StringRef IdentStr,
                                         bool SrcToBC = false) {
  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(IdentStr, Ident)) {
    return Status;
  }
  Triple = (Twine(Ident.Arch) + "-" + Ident.Vendor + "-" + Ident.OS).str();

  GPUArch = Twine(Ident.Processor).str();
  if (!Ident.Features.empty()) {
    GPUArch += ":" + join(Ident.Features, ":");
  }

  if (SrcToBC && getLanguage() == AMD_COMGR_LANGUAGE_HIP) {
    OffloadArch = (Twine("--offload-arch=") + GPUArch).str();
    Args.push_back(OffloadArch.c_str());
  } else {
    CPU = (Twine("-mcpu=") + GPUArch).str();
    Args.push_back("-target");
    Args.push_back(Triple.c_str());
    Args.push_back(CPU.c_str());
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::addCompilationFlags() {
  HIPIncludePath = (Twine(env::getHIPPath()) + "/include").str();
  // HIP headers depend on hsa.h which is in ROCM_DIR/include.
  ROCMIncludePath = (Twine(env::getROCMPath()) + "/include").str();
  ClangIncludePath =
      (Twine(env::getLLVMPath()) + "/lib/clang/" + CLANG_VERSION_STRING).str();
  ClangIncludePath2 = (Twine(env::getLLVMPath()) + "/lib/clang/" +
                       CLANG_VERSION_STRING + "/include")
                          .str();

  Args.push_back("-x");

  switch (ActionInfo->Language) {
  case AMD_COMGR_LANGUAGE_OPENCL_1_2:
    Args.push_back("cl");
    Args.push_back("-std=cl1.2");
    Args.push_back("-cl-no-stdinc");
    break;
  case AMD_COMGR_LANGUAGE_OPENCL_2_0:
    Args.push_back("cl");
    Args.push_back("-std=cl2.0");
    Args.push_back("-cl-no-stdinc");
    break;
  case AMD_COMGR_LANGUAGE_HIP:
    Args.push_back("hip");
    Args.push_back("-std=c++11");
    Args.push_back("-target");
    Args.push_back("x86_64-unknown-linux-gnu");
    Args.push_back("--cuda-device-only");
    Args.push_back("-nogpulib");
    Args.push_back("-isystem");
    Args.push_back(ROCMIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(HIPIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(ClangIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(ClangIncludePath2.c_str());
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::preprocessToSource() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

  Args.push_back("-E");

  return processFiles(AMD_COMGR_DATA_KIND_SOURCE, ".i");
}

amd_comgr_status_t AMDGPUCompiler::compileToBitcode(bool WithDeviceLibs) {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName, true)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

  Args.push_back("-c");
  Args.push_back("-emit-llvm");

#if _WIN32
  Args.push_back("-fshort-wchar");
#endif

  if (WithDeviceLibs) {
    llvm::SmallString<128> FakeRocmDir = TmpDir;
    path::append(FakeRocmDir, "rocm");
    llvm::SmallString<128> DeviceLibsDir = FakeRocmDir;
    path::append(DeviceLibsDir, "amdgcn", "bitcode");
    if (fs::create_directory(InputDir)) {
      return AMD_COMGR_STATUS_ERROR;
    }
    Args.push_back(Saver.save(Twine("--rocm-path=") + FakeRocmDir).data());
    NoGpuLib = false;

    for (auto DeviceLib : getDeviceLibraries()) {
      llvm::SmallString<128> DeviceLibPath = DeviceLibsDir;
      path::append(DeviceLibPath, std::get<0>(DeviceLib));
      if (auto Status = outputToFile(std::get<1>(DeviceLib), DeviceLibPath)) {
        return Status;
      }
    }
  }

  return processFiles(AMD_COMGR_DATA_KIND_BC, ".bc");
}

amd_comgr_status_t AMDGPUCompiler::compileToFatBin() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->Language != AMD_COMGR_LANGUAGE_HIP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // This is a workaround to support HIP OOP Fatbin Compilation
  CompileOOP = true;
  auto Status = processFiles(AMD_COMGR_DATA_KIND_FATBIN, ".fatbin");
  CompileOOP = false;

  return Status;
}

amd_comgr_status_t AMDGPUCompiler::linkBitcodeToBitcode() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  LLVMContext Context;
  Context.setDiagnosticHandler(
      std::make_unique<AMDGPUCompilerDiagnosticHandler>(this), true);

  auto Composite = std::make_unique<llvm::Module>("linked", Context);
  Linker L(*Composite);
  unsigned ApplicableFlags = Linker::Flags::None;

  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_BC) {
      continue;
    }

    SMDiagnostic SMDiag;
    // The data in Input outlives Mod, and the linker destructs Mod after
    // linking it into composite (i.e. ownership is not transferred to the
    // composite) so MemoryBuffer::getMemBuffer is sufficient.
    auto Mod =
        getLazyIRModule(MemoryBuffer::getMemBuffer(
                            StringRef(Input->Data, Input->Size), "", false),
                        SMDiag, Context, true);
    if (!Mod) {
      SMDiag.print(Input->Name, LogS, /* ShowColors */ false);
      return AMD_COMGR_STATUS_ERROR;
    }
    if (verifyModule(*Mod, &LogS)) {
      return AMD_COMGR_STATUS_ERROR;
    }
    if (L.linkInModule(std::move(Mod), ApplicableFlags)) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
  if (verifyModule(*Composite, &LogS)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  SmallString<0> OutBuf;
  BitcodeWriter Writer(OutBuf);
  Writer.writeModule(*Composite, false, nullptr, false, nullptr);
  Writer.writeSymtab();
  Writer.writeStrtab();

  amd_comgr_data_t OutputT;
  if (auto Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("linked.bc");
  Output->setData(OutBuf);

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

amd_comgr_status_t AMDGPUCompiler::codeGenBitcodeToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  Args.push_back("-c");

  return processFiles(AMD_COMGR_DATA_KIND_RELOCATABLE, ".o");
}

amd_comgr_status_t AMDGPUCompiler::codeGenBitcodeToAssembly() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  Args.push_back("-S");

  return processFiles(AMD_COMGR_DATA_KIND_SOURCE, ".s");
}

amd_comgr_status_t AMDGPUCompiler::assembleToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  Args.push_back("-c");

  return processFiles(AMD_COMGR_DATA_KIND_RELOCATABLE, ".o");
}

amd_comgr_status_t AMDGPUCompiler::linkToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Args.push_back(Option.c_str());
  }

  SmallVector<SmallString<128>, 128> Inputs;
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      continue;
    }

    Inputs.push_back(getFilePath(Input, InputDir));
    if (auto Status = outputToFile(Input, Inputs.back())) {
      return Status;
    }
    Args.push_back(Inputs.back().c_str());
  }

  amd_comgr_data_t OutputT;
  if (auto Status =
          amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("a.o");
  auto OutputFilePath = getFilePath(Output, OutputDir);
  Args.push_back("-o");
  Args.push_back(OutputFilePath.c_str());

  Args.push_back("-r");

  if (auto Status = linkWithLLD(Args, LogS, LogS)) {
    return Status;
  }

  if (auto Status = inputFromFile(Output, OutputFilePath)) {
    return Status;
  }

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

amd_comgr_status_t AMDGPUCompiler::linkToExecutable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Args.push_back(Option.c_str());
  }

  SmallVector<SmallString<128>, 128> Inputs;
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      continue;
    }

    Inputs.push_back(getFilePath(Input, InputDir));
    if (auto Status = outputToFile(Input, Inputs.back())) {
      return Status;
    }
    Args.push_back(Inputs.back().c_str());
  }

  amd_comgr_data_t OutputT;
  if (auto Status =
          amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("a.so");
  auto OutputFilePath = getFilePath(Output, OutputDir);
  Args.push_back("-o");
  Args.push_back(OutputFilePath.c_str());

  if (auto Status = executeInProcessDriver(Args)) {
    return Status;
  }

  if (auto Status = inputFromFile(Output, OutputFilePath)) {
    return Status;
  }

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

AMDGPUCompiler::AMDGPUCompiler(DataAction *ActionInfo, DataSet *InSet,
                               DataSet *OutSet, raw_ostream &LogS)
    : ActionInfo(ActionInfo), InSet(InSet), OutSetT(DataSet::convert(OutSet)),
      LogS(LogS) {
  initializeCommandLineArgs(Args);
}

AMDGPUCompiler::~AMDGPUCompiler() {
  if (!env::shouldSaveTemps()) {
    removeTmpDirs();
  }
}

} // namespace COMGR
