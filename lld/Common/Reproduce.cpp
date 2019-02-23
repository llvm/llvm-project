//===- Reproduce.cpp - Utilities for creating reproducers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Reproduce.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

// Makes a given pathname an absolute path first, and then remove
// beginning /. For example, "../foo.o" is converted to "home/john/foo.o",
// assuming that the current directory is "/home/john/bar".
// Returned string is a forward slash separated path even on Windows to avoid
// a mess with backslash-as-escape and backslash-as-path-separator.
std::string lld::relativeToRoot(StringRef Path) {
  SmallString<128> Abs = Path;
  if (fs::make_absolute(Abs))
    return Path;
  path::remove_dots(Abs, /*remove_dot_dot=*/true);

  // This is Windows specific. root_name() returns a drive letter
  // (e.g. "c:") or a UNC name (//net). We want to keep it as part
  // of the result.
  SmallString<128> Res;
  StringRef Root = path::root_name(Abs);
  if (Root.endswith(":"))
    Res = Root.drop_back();
  else if (Root.startswith("//"))
    Res = Root.substr(2);

  path::append(Res, path::relative_path(Abs));
  return path::convert_to_slash(Res);
}

// Quote a given string if it contains a space character.
std::string lld::quote(StringRef S) {
  if (S.contains(' '))
    return ("\"" + S + "\"").str();
  return S;
}

std::string lld::toString(const opt::Arg &Arg) {
  std::string K = Arg.getSpelling();
  if (Arg.getNumValues() == 0)
    return K;
  std::string V = quote(Arg.getValue());
  if (Arg.getOption().getRenderStyle() == opt::Option::RenderJoinedStyle)
    return K + V;
  return K + " " + V;
}
