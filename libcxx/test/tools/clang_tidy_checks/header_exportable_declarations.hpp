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
  void registerMatchers(clang::ast_matchers::MatchFinder*) override;
  void check(const clang::ast_matchers::MatchFinder::MatchResult&) override;

  enum class FileType { Header, ModulePartition, Module, Unknown };

private:
  llvm::StringRef filename_;
  FileType file_type_;
  llvm::StringRef extra_header_;
  std::set<std::string> decls_;
};
} // namespace libcpp
