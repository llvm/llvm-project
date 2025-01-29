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

using Decl =
    std::variant<struct RequiresDecl, struct HeaderDecl, struct UmbrellaDirDecl,
                 struct ModuleDecl, struct ExcludeDecl, struct ExportDecl,
                 struct ExportAsDecl, struct ExternModuleDecl, struct UseDecl,
                 struct LinkDecl, struct ConfigMacrosDecl, struct ConflictDecl>;

struct RequiresFeature {
  SourceLocation Location;
  StringRef Feature;
  bool RequiredState = true;
};

struct RequiresDecl {
  SourceLocation Location;
  std::vector<RequiresFeature> Features;
};

struct HeaderDecl {
  SourceLocation Location;
  StringRef Path;
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
  SourceLocation Location;
  StringRef Path;
};

struct ModuleDecl {
  SourceLocation Location; /// Points to the first keyword in the decl.
  ModuleId Id;
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
  SourceLocation Location;
  ModuleId Id;
  bool Wildcard;
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
  SourceLocation Location;
  StringRef Library;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Framework : 1;
};

struct ConfigMacrosDecl {
  SourceLocation Location;
  std::vector<StringRef> Macros;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Exhaustive : 1;
};

struct ConflictDecl {
  SourceLocation Location;
  ModuleId Id;
  StringRef Message;
};

using TopLevelDecl = std::variant<ModuleDecl, ExternModuleDecl>;

struct ModuleMapFile {
  std::vector<TopLevelDecl> Decls;
};

std::optional<ModuleMapFile> parseModuleMap(FileEntryRef File,
                                            SourceManager &SM,
                                            DiagnosticsEngine &Diags,
                                            bool IsSystem, unsigned *Offset);
void dumpModuleMapFile(ModuleMapFile &MMF, llvm::raw_ostream &out);

} // namespace modulemap
} // namespace clang

#endif
