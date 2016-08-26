//===- Reproduce.h - Utilities for creating reproducers ---------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_REPRODUCE_H
#define LLD_CORE_REPRODUCE_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

namespace llvm {

class raw_fd_ostream;

namespace opt { class Arg; }

}

namespace lld {

// This class creates a .cpio file for --reproduce (ELF) or /linkrepro (COFF).
//
// If "--reproduce foo" is given, we create a file "foo.cpio" and
// copy all input files to the archive, along with a response file
// to re-run the same command with the same inputs.
// It is useful for reporting issues to LLD developers.
//
// Cpio as a file format is a deliberate choice. It's standardized in
// POSIX and very easy to create. cpio command is available virtually
// on all Unix systems. See
// http://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html#tag_20_92_13_07
// for the format details.
class CpioFile {
public:
  static ErrorOr<CpioFile *> create(StringRef OutputPath);
  void append(StringRef Path, StringRef Data);

private:
  CpioFile(std::unique_ptr<llvm::raw_fd_ostream> OS, StringRef Basename);

  std::unique_ptr<llvm::raw_fd_ostream> OS;
  llvm::StringSet<> Seen;
  std::string Basename;
};

// Makes a given pathname an absolute path first, and then remove
// beginning /. For example, "../foo.o" is converted to "home/john/foo.o",
// assuming that the current directory is "/home/john/bar".
std::string relativeToRoot(StringRef Path);

// Quote a given string if it contains a space character.
std::string quote(StringRef S);

// Rewrite the given path if a file exists with that pathname, otherwise
// returns the original path.
std::string rewritePath(StringRef S);

// Returns the string form of the given argument.
std::string stringize(llvm::opt::Arg *Arg);

}

#endif
