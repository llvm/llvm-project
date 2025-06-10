//===- ModuleMapFile.h - Parsing and representation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_MODULEMAPFILE_H
#define LLVM_CLANG_LEX_MODULEMAPFILE_H

#include "clang/Basic/LLVM.h"
// TODO: Consider moving ModuleId to another header, parsing a modulemap file is
//   intended to not depend on anything about the clang::Module class.
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <variant>

namespace clang {

class DiagnosticsEngine;
class SourceManager;

namespace modulemap {

struct ExportDecl;

/// All declarations that can appear in a `module` declaration.
using Decl =
    std::variant<struct RequiresDecl, struct HeaderDecl, struct UmbrellaDirDecl,
                 struct ModuleDecl, struct ExcludeDecl, struct ExportDecl,
                 struct ExportAsDecl, struct ExternModuleDecl, struct UseDecl,
                 struct LinkDecl, struct ConfigMacrosDecl, struct ConflictDecl>;

struct RequiresFeature {
  StringRef Feature;
  SourceLocation Location;
  bool RequiredState = true; /// False if preceded by '!'.
};

struct RequiresDecl {
  SourceLocation Location;
  std::vector<RequiresFeature> Features;
};

struct HeaderDecl {
  StringRef Path;
  SourceLocation Location;
  SourceLocation PathLoc;
  std::optional<int64_t> Size;
  std::optional<int64_t> MTime;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Private : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Textual : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Umbrella : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Excluded : 1;
};

struct UmbrellaDirDecl {
  StringRef Path;
  SourceLocation Location;
};

struct ModuleDecl {
  ModuleId Id;
  SourceLocation Location; /// Points to the first keyword in the decl.
  ModuleAttributes Attrs;
  std::vector<Decl> Decls;

  LLVM_PREFERRED_TYPE(bool)
  unsigned Explicit : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Framework : 1;
};

struct ExcludeDecl {
  SourceLocation Location;
  StringRef Module;
};

struct ExportDecl {
  ModuleId Id;
  SourceLocation Location;
  bool Wildcard; /// True if the last element of the ModuleId is '*'.
};

struct ExportAsDecl {
  SourceLocation Location;
  ModuleId Id;
};

struct ExternModuleDecl {
  SourceLocation Location;
  ModuleId Id;
  StringRef Path;
};

struct UseDecl {
  SourceLocation Location;
  ModuleId Id;
};

struct LinkDecl {
  StringRef Library;
  SourceLocation Location;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Framework : 1;
};

struct ConfigMacrosDecl {
  std::vector<StringRef> Macros;
  SourceLocation Location;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Exhaustive : 1;
};

struct ConflictDecl {
  SourceLocation Location;
  ModuleId Id;
  StringRef Message;
};

using TopLevelDecl = std::variant<ModuleDecl, ExternModuleDecl>;

/// Represents the parsed form of a module map file.
///
/// This holds many reference types (StringRef, SourceLocation, etc.) whose
/// lifetimes are bound by the SourceManager and FileManager used.
struct ModuleMapFile {
  /// The FileID used to parse this module map. This is always a local ID.
  FileID ID;

  /// The directory in which the module map was discovered. Declarations in
  /// the module map are relative to this directory.
  OptionalDirectoryEntryRef Dir;

  /// Beginning of the file, used for moduleMapFileRead callback.
  SourceLocation Start;

  bool IsSystem;
  std::vector<TopLevelDecl> Decls;

  void dump(llvm::raw_ostream &out) const;
};

/// Parse a module map file into an in memory representation.
///
/// \param ID a valid local FileID.
/// \param Dir the directory in which this module map was found.
/// \param SM the SourceManager for \a ID.
/// \param Diags where to send the diagnostics.
/// \param IsSystem was this module map found in a system search path.
/// \param Offset optional offset into the buffer associated with \a ID. This is
///               used for handling `#pragma clang module build`. Set to the end
///               of the module map on return.
///
/// \returns The parsed ModuleMapFile if successful, std::nullopt otherwise.
std::optional<ModuleMapFile>
parseModuleMap(FileID ID, clang::DirectoryEntryRef Dir, SourceManager &SM,
               DiagnosticsEngine &Diags, bool IsSystem, unsigned *Offset);

} // namespace modulemap
} // namespace clang

#endif
