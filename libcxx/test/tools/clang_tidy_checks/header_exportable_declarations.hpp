//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/ADT/StringRef.h"

#include <fstream>
#include <set>
#include <string>

namespace libcpp {
class header_exportable_declarations : public clang::tidy::ClangTidyCheck {
public:
  explicit header_exportable_declarations(llvm::StringRef, clang::tidy::ClangTidyContext*);
  ~header_exportable_declarations();
  void registerMatchers(clang::ast_matchers::MatchFinder*) override;
  void check(const clang::ast_matchers::MatchFinder::MatchResult&) override;

  enum class FileType {
    // std module specific
    Header,
    CompatModulePartition,
    Module,
    // std.compat module specific
    CHeader,
    ModulePartition,
    CompatModule,
    // invalid value
    Unknown
  };

private:
  llvm::StringRef filename_;
  FileType file_type_;
  llvm::StringRef extra_header_;
  std::set<std::string> decls_;
  std::set<std::string> global_decls_;

  // The named declarations in .h C headers are "tricky". On POSIX
  // systems these headers contain POSIX specific functions that do not
  // use a reserved name. For example, fmemopen is provided by stdio.h.
  // We filter the names that should be provided by the headers as follows:
  // - record all named declarations the global namespace
  // - wait until the header is completely processed
  // - every named declaration in the global namespace that has a matching
  //   "export" in the std namespace is exported.
  //
  // The only place where we can do the above while ensuring that all
  // the declarations in the header have been seen is in the clang tidy
  // plugin's destructor.
  //
  // It is possible to skip some declarations in the std namespace,
  // these are added to decls_ before processing. To differentiate
  // between a skipped declaration and a real declaration the skipped
  // declarations are recorded in an extra variable.
  std::set<std::string> skip_decls_;
};
} // namespace libcpp
