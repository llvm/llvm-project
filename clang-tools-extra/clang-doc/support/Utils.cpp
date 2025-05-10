//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;

llvm::SmallString<128> appendPathNative(StringRef Path, StringRef Asset) {
  llvm::SmallString<128> Default;
  llvm::sys::path::native(Path, Default);
  llvm::sys::path::append(Default, Asset);
  return Default;
}

void getMustacheHtmlFiles(StringRef AssetsPath,
                          clang::doc::ClangDocContext &CDCtx) {
  assert(!AssetsPath.empty());
  assert(llvm::sys::fs::is_directory(AssetsPath));

  llvm::SmallString<128> DefaultStylesheet =
      appendPathNative(AssetsPath, "clang-doc-mustache.css");
  llvm::SmallString<128> NamespaceTemplate =
      appendPathNative(AssetsPath, "namespace-template.mustache");
  llvm::SmallString<128> ClassTemplate =
      appendPathNative(AssetsPath, "class-template.mustache");
  llvm::SmallString<128> EnumTemplate =
      appendPathNative(AssetsPath, "enum-template.mustache");
  llvm::SmallString<128> FunctionTemplate =
      appendPathNative(AssetsPath, "function-template.mustache");
  llvm::SmallString<128> CommentTemplate =
      appendPathNative(AssetsPath, "comments-template.mustache");
  llvm::SmallString<128> IndexJS =
      appendPathNative(AssetsPath, "mustache-index.js");

  CDCtx.JsScripts.insert(CDCtx.JsScripts.begin(), IndexJS.c_str());
  CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(),
                               DefaultStylesheet.str().str());
  CDCtx.MustacheTemplates.insert(
      {"namespace-template", NamespaceTemplate.c_str()});
  CDCtx.MustacheTemplates.insert({"class-template", ClassTemplate.c_str()});
  CDCtx.MustacheTemplates.insert({"enum-template", EnumTemplate.c_str()});
  CDCtx.MustacheTemplates.insert(
      {"function-template", FunctionTemplate.c_str()});
  CDCtx.MustacheTemplates.insert(
      {"comments-template", CommentTemplate.c_str()});
}
