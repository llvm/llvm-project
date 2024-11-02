//===-- Options.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Options.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/InstallAPI/FileList.h"
#include "clang/InstallAPI/InstallAPIDiagnostic.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TextAPI/DylibReader.h"
#include "llvm/TextAPI/TextAPIWriter.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::MachO;

namespace drv = clang::driver::options;

namespace clang {
namespace installapi {

/// Create prefix string literals used in InstallAPIOpts.td.
#define PREFIX(NAME, VALUE)                                                    \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE;                  \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(                   \
      NAME##_init, std::size(NAME##_init) - 1);
#include "InstallAPIOpts.inc"
#undef PREFIX

static constexpr const llvm::StringLiteral PrefixTable_init[] =
#define PREFIX_UNION(VALUES) VALUES
#include "InstallAPIOpts.inc"
#undef PREFIX_UNION
    ;
static constexpr const ArrayRef<StringLiteral>
    PrefixTable(PrefixTable_init, std::size(PrefixTable_init) - 1);

/// Create table mapping all options defined in InstallAPIOpts.td.
static constexpr OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, METAVAR, VALUES)                   \
  {PREFIX, NAME,  HELPTEXT,   METAVAR,     OPT_##ID,    Option::KIND##Class,   \
   PARAM,  FLAGS, VISIBILITY, OPT_##GROUP, OPT_##ALIAS, ALIASARGS,             \
   VALUES},
#include "InstallAPIOpts.inc"
#undef OPTION
};

namespace {

/// \brief Create OptTable class for parsing actual command line arguments.
class DriverOptTable : public opt::PrecomputedOptTable {
public:
  DriverOptTable() : PrecomputedOptTable(InfoTable, PrefixTable) {}
};

} // end anonymous namespace.

static llvm::opt::OptTable *createDriverOptTable() {
  return new DriverOptTable();
}

bool Options::processDriverOptions(InputArgList &Args) {
  // Handle inputs.
  llvm::append_range(DriverOpts.FileLists,
                     Args.getAllArgValues(drv::OPT_INPUT));

  // Handle output.
  SmallString<PATH_MAX> OutputPath;
  if (auto *Arg = Args.getLastArg(drv::OPT_o)) {
    OutputPath = Arg->getValue();
    if (OutputPath != "-")
      FM->makeAbsolutePath(OutputPath);
    DriverOpts.OutputPath = std::string(OutputPath);
  }
  if (DriverOpts.OutputPath.empty()) {
    Diags->Report(diag::err_no_output_file);
    return false;
  }

  // Do basic error checking first for mixing -target and -arch options.
  auto *ArgArch = Args.getLastArgNoClaim(drv::OPT_arch);
  auto *ArgTarget = Args.getLastArgNoClaim(drv::OPT_target);
  auto *ArgTargetVariant =
      Args.getLastArgNoClaim(drv::OPT_darwin_target_variant_triple);
  if (ArgArch && (ArgTarget || ArgTargetVariant)) {
    Diags->Report(clang::diag::err_drv_argument_not_allowed_with)
        << ArgArch->getAsString(Args)
        << (ArgTarget ? ArgTarget : ArgTargetVariant)->getAsString(Args);
    return false;
  }

  auto *ArgMinTargetOS = Args.getLastArgNoClaim(drv::OPT_mtargetos_EQ);
  if ((ArgTarget || ArgTargetVariant) && ArgMinTargetOS) {
    Diags->Report(clang::diag::err_drv_cannot_mix_options)
        << ArgTarget->getAsString(Args) << ArgMinTargetOS->getAsString(Args);
    return false;
  }

  // Capture target triples first.
  if (ArgTarget) {
    for (const Arg *A : Args.filtered(drv::OPT_target)) {
      A->claim();
      llvm::Triple TargetTriple(A->getValue());
      Target TAPITarget = Target(TargetTriple);
      if ((TAPITarget.Arch == AK_unknown) ||
          (TAPITarget.Platform == PLATFORM_UNKNOWN)) {
        Diags->Report(clang::diag::err_drv_unsupported_opt_for_target)
            << "installapi" << TargetTriple.str();
        return false;
      }
      DriverOpts.Targets[TAPITarget] = TargetTriple;
    }
  }

  DriverOpts.Verbose = Args.hasArgNoClaim(drv::OPT_v);

  return true;
}

bool Options::processLinkerOptions(InputArgList &Args) {
  // Handle required arguments.
  if (const Arg *A = Args.getLastArg(drv::OPT_install__name))
    LinkerOpts.InstallName = A->getValue();
  if (LinkerOpts.InstallName.empty()) {
    Diags->Report(diag::err_no_install_name);
    return false;
  }

  // Defaulted or optional arguments.
  if (auto *Arg = Args.getLastArg(drv::OPT_current__version))
    LinkerOpts.CurrentVersion.parse64(Arg->getValue());

  if (auto *Arg = Args.getLastArg(drv::OPT_compatibility__version))
    LinkerOpts.CompatVersion.parse64(Arg->getValue());

  LinkerOpts.IsDylib = Args.hasArg(drv::OPT_dynamiclib);

  LinkerOpts.AppExtensionSafe = Args.hasFlag(
      drv::OPT_fapplication_extension, drv::OPT_fno_application_extension,
      /*Default=*/LinkerOpts.AppExtensionSafe);

  if (::getenv("LD_NO_ENCRYPT") != nullptr)
    LinkerOpts.AppExtensionSafe = true;

  if (::getenv("LD_APPLICATION_EXTENSION_SAFE") != nullptr)
    LinkerOpts.AppExtensionSafe = true;
  return true;
}

bool Options::processFrontendOptions(InputArgList &Args) {
  // Do not claim any arguments, as they will be passed along for CC1
  // invocations.
  if (auto *A = Args.getLastArgNoClaim(drv::OPT_x)) {
    FEOpts.LangMode = llvm::StringSwitch<clang::Language>(A->getValue())
                          .Case("c", clang::Language::C)
                          .Case("c++", clang::Language::CXX)
                          .Case("objective-c", clang::Language::ObjC)
                          .Case("objective-c++", clang::Language::ObjCXX)
                          .Default(clang::Language::Unknown);

    if (FEOpts.LangMode == clang::Language::Unknown) {
      Diags->Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
      return false;
    }
  }
  for (auto *A : Args.filtered(drv::OPT_ObjC, drv::OPT_ObjCXX)) {
    if (A->getOption().matches(drv::OPT_ObjC))
      FEOpts.LangMode = clang::Language::ObjC;
    else
      FEOpts.LangMode = clang::Language::ObjCXX;
  }

  return true;
}

std::vector<const char *>
Options::processAndFilterOutInstallAPIOptions(ArrayRef<const char *> Args) {
  std::unique_ptr<llvm::opt::OptTable> Table;
  Table.reset(createDriverOptTable());

  unsigned MissingArgIndex, MissingArgCount;
  auto ParsedArgs = Table->ParseArgs(Args.slice(1), MissingArgIndex,
                                     MissingArgCount, Visibility());

  // Capture InstallAPI only driver options.
  DriverOpts.Demangle = ParsedArgs.hasArg(OPT_demangle);

  if (auto *A = ParsedArgs.getLastArg(OPT_filetype)) {
    DriverOpts.OutFT = TextAPIWriter::parseFileType(A->getValue());
    if (DriverOpts.OutFT == FileType::Invalid) {
      Diags->Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(ParsedArgs) << A->getValue();
      return {};
    }
  }

  if (const Arg *A = ParsedArgs.getLastArg(OPT_verify_mode_EQ)) {
    DriverOpts.VerifyMode =
        StringSwitch<VerificationMode>(A->getValue())
            .Case("ErrorsOnly", VerificationMode::ErrorsOnly)
            .Case("ErrorsAndWarnings", VerificationMode::ErrorsAndWarnings)
            .Case("Pedantic", VerificationMode::Pedantic)
            .Default(VerificationMode::Invalid);

    if (DriverOpts.VerifyMode == VerificationMode::Invalid) {
      Diags->Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(ParsedArgs) << A->getValue();
      return {};
    }
  }

  if (const Arg *A = ParsedArgs.getLastArg(OPT_verify_against))
    DriverOpts.DylibToVerify = A->getValue();

  /// Any unclaimed arguments should be forwarded to the clang driver.
  std::vector<const char *> ClangDriverArgs(ParsedArgs.size());
  for (const Arg *A : ParsedArgs) {
    if (A->isClaimed())
      continue;
    llvm::copy(A->getValues(), std::back_inserter(ClangDriverArgs));
  }
  return ClangDriverArgs;
}

Options::Options(DiagnosticsEngine &Diag, FileManager *FM,
                 ArrayRef<const char *> Args, const StringRef ProgName)
    : Diags(&Diag), FM(FM) {

  // First process InstallAPI specific options.
  auto DriverArgs = processAndFilterOutInstallAPIOptions(Args);
  if (Diags->hasErrorOccurred())
    return;

  // Set up driver to parse remaining input arguments.
  clang::driver::Driver Driver(ProgName, llvm::sys::getDefaultTargetTriple(),
                               *Diags, "clang installapi tool");
  auto TargetAndMode =
      clang::driver::ToolChain::getTargetAndModeFromProgramName(ProgName);
  Driver.setTargetAndMode(TargetAndMode);
  bool HasError = false;
  llvm::opt::InputArgList ArgList =
      Driver.ParseArgStrings(DriverArgs, /*UseDriverMode=*/true, HasError);
  if (HasError)
    return;
  Driver.setCheckInputsExist(false);

  if (!processDriverOptions(ArgList))
    return;

  if (!processLinkerOptions(ArgList))
    return;

  if (!processFrontendOptions(ArgList))
    return;

  /// Force cc1 options that should always be on.
  FrontendArgs = {"-fsyntax-only", "-Wprivate-extern"};

  /// Any unclaimed arguments should be handled by invoking the clang frontend.
  for (const Arg *A : ArgList) {
    if (A->isClaimed())
      continue;
    FrontendArgs.emplace_back(A->getSpelling());
    llvm::copy(A->getValues(), std::back_inserter(FrontendArgs));
  }
}

InstallAPIContext Options::createContext() {
  InstallAPIContext Ctx;
  Ctx.FM = FM;
  Ctx.Diags = Diags;

  // InstallAPI requires two level namespacing.
  Ctx.BA.TwoLevelNamespace = true;

  Ctx.BA.InstallName = LinkerOpts.InstallName;
  Ctx.BA.CurrentVersion = LinkerOpts.CurrentVersion;
  Ctx.BA.CompatVersion = LinkerOpts.CompatVersion;
  Ctx.BA.AppExtensionSafe = LinkerOpts.AppExtensionSafe;
  Ctx.FT = DriverOpts.OutFT;
  Ctx.OutputLoc = DriverOpts.OutputPath;
  Ctx.LangMode = FEOpts.LangMode;

  // Process inputs.
  for (const std::string &ListPath : DriverOpts.FileLists) {
    auto Buffer = FM->getBufferForFile(ListPath);
    if (auto Err = Buffer.getError()) {
      Diags->Report(diag::err_cannot_open_file) << ListPath << Err.message();
      return Ctx;
    }
    if (auto Err = FileListReader::loadHeaders(std::move(Buffer.get()),
                                               Ctx.InputHeaders)) {
      Diags->Report(diag::err_cannot_open_file) << ListPath << std::move(Err);
      return Ctx;
    }
  }

  // Parse binary dylib and initialize verifier.
  if (DriverOpts.DylibToVerify.empty()) {
    Ctx.Verifier = std::make_unique<DylibVerifier>();
    return Ctx;
  }

  auto Buffer = FM->getBufferForFile(DriverOpts.DylibToVerify);
  if (auto Err = Buffer.getError()) {
    Diags->Report(diag::err_cannot_open_file)
        << DriverOpts.DylibToVerify << Err.message();
    return Ctx;
  }

  DylibReader::ParseOption PO;
  PO.Undefineds = false;
  Expected<Records> Slices =
      DylibReader::readFile((*Buffer)->getMemBufferRef(), PO);
  if (auto Err = Slices.takeError()) {
    Diags->Report(diag::err_cannot_open_file) << DriverOpts.DylibToVerify;
    return Ctx;
  }

  Ctx.Verifier = std::make_unique<DylibVerifier>(
      std::move(*Slices), Diags, DriverOpts.VerifyMode, DriverOpts.Demangle);
  return Ctx;
}

} // namespace installapi
} // namespace clang
