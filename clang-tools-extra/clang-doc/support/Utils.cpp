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

SmallString<128> appendPathNative(StringRef Base, StringRef Path) {
  SmallString<128> Default;
  sys::path::native(Base, Default);
  sys::path::append(Default, Path);
  return Default;
}

SmallString<128> appendPathPosix(StringRef Base, StringRef Path) {
  SmallString<128> Default;
  sys::path::native(Base, Default, sys::path::Style::posix);
  sys::path::append(Default, Path);
  return Default;
}

void getMustacheHtmlFiles(StringRef AssetsPath,
                          clang::doc::ClangDocContext &CDCtx) {
  assert(!AssetsPath.empty());
  assert(sys::fs::is_directory(AssetsPath));

  SmallString<128> DefaultStylesheet =
      appendPathPosix(AssetsPath, "clang-doc-mustache.css");
  SmallString<128> NamespaceTemplate =
      appendPathPosix(AssetsPath, "namespace-template.mustache");
  SmallString<128> ClassTemplate =
      appendPathPosix(AssetsPath, "class-template.mustache");
  SmallString<128> EnumTemplate =
      appendPathPosix(AssetsPath, "enum-template.mustache");
  SmallString<128> FunctionTemplate =
      appendPathPosix(AssetsPath, "function-template.mustache");
  SmallString<128> CommentTemplate =
      appendPathPosix(AssetsPath, "comment-template.mustache");
  SmallString<128> IndexJS = appendPathPosix(AssetsPath, "mustache-index.js");

  CDCtx.JsScripts.insert(CDCtx.JsScripts.begin(), IndexJS.c_str());
  CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(),
                               DefaultStylesheet.c_str());
  CDCtx.MustacheTemplates.insert(
      {"namespace-template", NamespaceTemplate.c_str()});
  CDCtx.MustacheTemplates.insert({"class-template", ClassTemplate.c_str()});
  CDCtx.MustacheTemplates.insert({"enum-template", EnumTemplate.c_str()});
  CDCtx.MustacheTemplates.insert(
      {"function-template", FunctionTemplate.c_str()});
  CDCtx.MustacheTemplates.insert({"comment-template", CommentTemplate.c_str()});
}
