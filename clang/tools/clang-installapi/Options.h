//===--- clang-installapi/Options.h - Options -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_INSTALLAPI_OPTIONS_H
#define LLVM_CLANG_TOOLS_CLANG_INSTALLAPI_OPTIONS_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/InstallAPI/Context.h"
#include "clang/InstallAPI/DylibVerifier.h"
#include "clang/InstallAPI/MachO.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Triple.h"
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace installapi {

struct DriverOptions {
  /// \brief Path to input file lists (JSON).
  llvm::MachO::PathSeq FileLists;

  /// \brief Path to public umbrella header.
  std::string PublicUmbrellaHeader;

  /// \brief Path to private umbrella header.
  std::string PrivateUmbrellaHeader;

  /// \brief Path to project umbrella header.
  std::string ProjectUmbrellaHeader;

  /// \brief Paths of extra public headers.
  PathSeq ExtraPublicHeaders;

  /// \brief Paths of extra private headers.
  PathSeq ExtraPrivateHeaders;

  /// \brief Paths of extra project headers.
  PathSeq ExtraProjectHeaders;

  /// \brief List of excluded public headers.
  PathSeq ExcludePublicHeaders;

  /// \brief List of excluded private headers.
  PathSeq ExcludePrivateHeaders;

  /// \brief List of excluded project headers.
  PathSeq ExcludeProjectHeaders;

  /// \brief Mappings of target triples & tapi targets to build for.
  std::map<llvm::MachO::Target, llvm::Triple> Targets;

  /// \brief Path to binary dylib for comparing.
  std::string DylibToVerify;

  /// \brief Output path.
  std::string OutputPath;

  /// \brief File encoding to print.
  FileType OutFT = FileType::TBD_V5;

  /// \brief Verification mode for comparing symbols.
  VerificationMode VerifyMode = VerificationMode::Pedantic;

  /// \brief Print demangled symbols when reporting errors.
  bool Demangle = false;

  /// \brief Print verbose output.
  bool Verbose = false;
};

struct LinkerOptions {
  /// \brief The install name to use for the dynamic library.
  std::string InstallName;

  /// \brief The current version to use for the dynamic library.
  PackedVersion CurrentVersion;

  /// \brief The compatibility version to use for the dynamic library.
  PackedVersion CompatVersion;

  /// \brief Is application extension safe.
  bool AppExtensionSafe = false;

  /// \brief Set if we should scan for a dynamic library and not a framework.
  bool IsDylib = false;
};

struct FrontendOptions {
  /// \brief The language mode to parse headers in.
  Language LangMode = Language::ObjC;
};

class Options {
private:
  bool processDriverOptions(llvm::opt::InputArgList &Args);
  bool processLinkerOptions(llvm::opt::InputArgList &Args);
  bool processFrontendOptions(llvm::opt::InputArgList &Args);
  std::vector<const char *>
  processAndFilterOutInstallAPIOptions(ArrayRef<const char *> Args);

public:
  /// The various options grouped together.
  DriverOptions DriverOpts;
  LinkerOptions LinkerOpts;
  FrontendOptions FEOpts;

  Options() = delete;

  /// \brief Create InstallAPIContext from processed options.
  InstallAPIContext createContext();

  /// \brief Constructor for options.
  Options(clang::DiagnosticsEngine &Diag, FileManager *FM,
          ArrayRef<const char *> Args, const StringRef ProgName);

  /// \brief Get CC1 arguments after extracting out the irrelevant
  /// ones.
  std::vector<std::string> &getClangFrontendArgs() { return FrontendArgs; }

private:
  bool addFilePaths(llvm::opt::InputArgList &Args, PathSeq &Headers,
                    llvm::opt::OptSpecifier ID);

  DiagnosticsEngine *Diags;
  FileManager *FM;
  std::vector<std::string> FrontendArgs;
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, METAVAR, VALUES)                   \
  OPT_##ID,
#include "InstallAPIOpts.inc"
  LastOption
#undef OPTION
};

} // namespace installapi
} // namespace clang
#endif
