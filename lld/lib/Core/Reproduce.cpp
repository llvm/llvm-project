//===- Reproduce.cpp - Utilities for creating reproducers -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Reproduce.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"

using namespace lld;
using namespace llvm;
using namespace sys;

CpioFile::CpioFile(std::unique_ptr<raw_fd_ostream> OS, StringRef S)
    : OS(std::move(OS)), Basename(S) {}

ErrorOr<CpioFile *> CpioFile::create(StringRef OutputPath) {
  std::string Path = (OutputPath + ".cpio").str();
  std::error_code EC;
  auto OS = llvm::make_unique<raw_fd_ostream>(Path, EC, sys::fs::F_None);
  if (EC)
    return EC;
  return new CpioFile(std::move(OS), path::filename(OutputPath));
}

static void writeMember(raw_fd_ostream &OS, StringRef Path, StringRef Data) {
  // The c_dev/c_ino pair should be unique according to the spec,
  // but no one seems to care.
  OS << "070707";                        // c_magic
  OS << "000000";                        // c_dev
  OS << "000000";                        // c_ino
  OS << "100664";                        // c_mode: C_ISREG | rw-rw-r--
  OS << "000000";                        // c_uid
  OS << "000000";                        // c_gid
  OS << "000001";                        // c_nlink
  OS << "000000";                        // c_rdev
  OS << "00000000000";                   // c_mtime
  OS << format("%06o", Path.size() + 1); // c_namesize
  OS << format("%011o", Data.size());    // c_filesize
  OS << Path << '\0';                    // c_name
  OS << Data;                            // c_filedata
}

void CpioFile::append(StringRef Path, StringRef Data) {
  if (!Seen.insert(Path).second)
    return;

  // Construct an in-archive filename so that /home/foo/bar is stored
  // as baz/home/foo/bar where baz is the basename of the output file.
  // (i.e. in that case we are creating baz.cpio.)
  SmallString<128> Fullpath;
  path::append(Fullpath, Basename, Path);

  // Use unix path separators so the cpio can be extracted on both unix and
  // windows.
  std::replace(Fullpath.begin(), Fullpath.end(), '\\', '/');

  writeMember(*OS, Fullpath, Data);

  // Print the trailer and seek back.
  // This way we have a valid archive if we crash.
  uint64_t Pos = OS->tell();
  writeMember(*OS, "TRAILER!!!", "");
  OS->seek(Pos);
}

// Makes a given pathname an absolute path first, and then remove
// beginning /. For example, "../foo.o" is converted to "home/john/foo.o",
// assuming that the current directory is "/home/john/bar".
std::string lld::relativeToRoot(StringRef Path) {
  SmallString<128> Abs = Path;
  if (sys::fs::make_absolute(Abs))
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

  return Res.str();
}

// Quote a given string if it contains a space character.
std::string lld::quote(StringRef S) {
  if (S.find(' ') == StringRef::npos)
    return S;
  return ("\"" + S + "\"").str();
}

std::string lld::rewritePath(StringRef S) {
  if (fs::exists(S))
    return relativeToRoot(S);
  return S;
}

std::string lld::stringize(opt::Arg *Arg) {
  std::string K = Arg->getSpelling();
  if (Arg->getNumValues() == 0)
    return K;
  std::string V = quote(Arg->getValue());
  if (Arg->getOption().getRenderStyle() == opt::Option::RenderJoinedStyle)
    return K + V;
  return K + " " + V;
}
