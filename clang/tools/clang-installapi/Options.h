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
#include "clang/Frontend/FrontendOptions.h"
#include "clang/InstallAPI/Context.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/PackedVersion.h"
#include "llvm/TextAPI/Platform.h"
#include "llvm/TextAPI/Target.h"
#include "llvm/TextAPI/Utils.h"
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace installapi {
using Macro = std::pair<std::string, bool /*isUndef*/>;

struct DriverOptions {
  /// \brief Path to input file lists (JSON).
  llvm::MachO::PathSeq FileLists;

  /// \brief Mappings of target triples & tapi targets to build for.
  std::map<llvm::MachO::Target, llvm::Triple> Targets;

  /// \brief Output path.
  std::string OutputPath;

  /// \brief File encoding to print.
  llvm::MachO::FileType OutFT = llvm::MachO::FileType::TBD_V5;

  /// \brief Print verbose output.
  bool Verbose = false;
};

struct LinkerOptions {
  /// \brief The install name to use for the dynamic library.
  std::string InstallName;

  /// \brief The current version to use for the dynamic library.
  llvm::MachO::PackedVersion CurrentVersion;

  /// \brief Is application extension safe.
  bool AppExtensionSafe = false;

  /// \brief Set if we should scan for a dynamic library and not a framework.
  bool IsDylib = false;
};

class Options {
private:
  bool processDriverOptions(llvm::opt::InputArgList &Args);
  bool processLinkerOptions(llvm::opt::InputArgList &Args);

public:
  /// The various options grouped together.
  DriverOptions DriverOpts;
  LinkerOptions LinkerOpts;

  Options() = delete;

  /// \brief Create InstallAPIContext from processed options.
  InstallAPIContext createContext();

  /// \brief Constructor for options.
  Options(clang::DiagnosticsEngine &Diag, FileManager *FM,
          llvm::opt::InputArgList &Args);

  /// \brief Get CC1 arguments after extracting out the irrelevant
  /// ones.
  std::vector<std::string> &getClangFrontendArgs() { return FrontendArgs; }

private:
  DiagnosticsEngine *Diags;
  FileManager *FM;
  std::vector<std::string> FrontendArgs;
};

} // namespace installapi
} // namespace clang
#endif
