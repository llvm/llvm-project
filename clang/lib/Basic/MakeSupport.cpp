//===-- MakeSuport.cpp --------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/MakeSupport.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

void clang::quoteMakeTarget(StringRef Target, SmallVectorImpl<char> &Res) {
  for (unsigned i = 0, e = Target.size(); i != e; ++i) {
    switch (Target[i]) {
    case ' ':
    case '\t':
      // Escape the preceding backslashes
      for (int j = i - 1; j >= 0 && Target[j] == '\\'; --j)
        Res.push_back('\\');

      // Escape the space/tab
      Res.push_back('\\');
      break;
    case '$':
      Res.push_back('$');
      break;
    case '#':
      Res.push_back('\\');
      break;
    default:
      break;
    }

    Res.push_back(Target[i]);
  }
}

/// Print the filename, with escaping or quoting that accommodates the three
/// most likely tools that use dependency files: GNU Make, BSD Make, and
/// NMake/Jom.
///
/// BSD Make is the simplest case: It does no escaping at all.  This means
/// characters that are normally delimiters, i.e. space and # (the comment
/// character) simply aren't supported in filenames.
///
/// GNU Make does allow space and # in filenames, but to avoid being treated
/// as a delimiter or comment, these must be escaped with a backslash. Because
/// backslash is itself the escape character, if a backslash appears in a
/// filename, it should be escaped as well.  (As a special case, $ is escaped
/// as $$, which is the normal Make way to handle the $ character.)
/// For compatibility with BSD Make and historical practice, if GNU Make
/// un-escapes characters in a filename but doesn't find a match, it will
/// retry with the unmodified original string.
///
/// GCC tries to accommodate both Make formats by escaping any space or #
/// characters in the original filename, but not escaping backslashes.  The
/// apparent intent is so that filenames with backslashes will be handled
/// correctly by BSD Make, and by GNU Make in its fallback mode of using the
/// unmodified original string; filenames with # or space characters aren't
/// supported by BSD Make at all, but will be handled correctly by GNU Make
/// due to the escaping.
///
/// A corner case that GCC gets only partly right is when the original filename
/// has a backslash immediately followed by space or #.  GNU Make would expect
/// this backslash to be escaped; however GCC escapes the original backslash
/// only when followed by space, not #.  It will therefore take a dependency
/// from a directive such as
///     #include "a\ b\#c.h"
/// and emit it as
///     a\\\ b\\#c.h
/// which GNU Make will interpret as
///     a\ b\
/// followed by a comment. Failing to find this file, it will fall back to the
/// original string, which probably doesn't exist either; in any case it won't
/// find
///     a\ b\#c.h
/// which is the actual filename specified by the include directive.
///
/// Clang does what GCC does, rather than what GNU Make expects.
///
/// NMake/Jom has a different set of scary characters, but wraps filespecs in
/// double-quotes to avoid misinterpreting them; see
/// https://msdn.microsoft.com/en-us/library/dd9y37ha.aspx for NMake info,
/// https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx
/// for Windows file-naming info.
static void printFilename(llvm::raw_ostream &OS, llvm::StringRef Filename,
                          clang::DependencyOutputFormat OutputFormat) {
  // Convert filename to platform native path
  llvm::SmallString<256> NativePath;
  llvm::sys::path::native(Filename, NativePath);

  if (OutputFormat == clang::DependencyOutputFormat::NMake) {
    // Add quotes if needed. These are the characters listed as "special" to
    // NMake, that are legal in a Windows filespec, and that could cause
    // misinterpretation of the dependency string.
    if (NativePath.find_first_of(" #${}^!") != llvm::StringRef::npos)
      OS << '\"' << NativePath << '\"';
    else
      OS << NativePath;
    return;
  }
  assert(OutputFormat == clang::DependencyOutputFormat::Make);
  for (unsigned i = 0, e = NativePath.size(); i != e; ++i) {
    if (NativePath[i] == '#') // Handle '#' the broken gcc way.
      OS << '\\';
    else if (NativePath[i] == ' ') { // Handle space correctly.
      OS << '\\';
      unsigned j = i;
      while (j > 0 && NativePath[--j] == '\\')
        OS << '\\';
    } else if (NativePath[i] == '$') // $ is escaped by $$.
      OS << '$';
    OS << NativePath[i];
  }
}

void clang::printMakeDependencyFile(llvm::raw_ostream &OS,
                                    llvm::ArrayRef<std::string> Targets,
                                    llvm::ArrayRef<std::string> Files,
                                    DependencyOutputFormat Format,
                                    bool PhonyTargets,
                                    unsigned InputFileIndex) {
  // Write out the dependency targets, trying to avoid overly long
  // lines when possible. We try our best to emit exactly the same
  // dependency file as GCC>=10, assuming the included files are the
  // same.
  const unsigned MaxColumns = 75;
  unsigned Columns = 0;

  for (StringRef Target : Targets) {
    unsigned N = Target.size();
    if (Columns == 0) {
      Columns += N;
    } else if (Columns + N + 2 > MaxColumns) {
      Columns = N + 2;
      OS << " \\\n  ";
    } else {
      Columns += N + 1;
      OS << ' ';
    }
    // Targets already quoted as needed.
    OS << Target;
  }

  OS << ':';
  Columns += 1;

  // Now add each dependency in the order it was seen, but avoiding
  // duplicates.
  for (StringRef File : Files) {
    if (File == "<stdin>")
      continue;
    // Start a new line if this would exceed the column limit. Make
    // sure to leave space for a trailing " \" in case we need to
    // break the line on the next iteration.
    unsigned N = File.size();
    if (Columns + (N + 1) + 2 > MaxColumns) {
      OS << " \\\n ";
      Columns = 2;
    }
    OS << ' ';
    printFilename(OS, File, Format);
    Columns += N + 1;
  }
  OS << '\n';

  // Create phony targets if requested.
  if (PhonyTargets && !Files.empty()) {
    unsigned Index = 0;
    for (auto I = Files.begin(), E = Files.end(); I != E; ++I) {
      if (Index++ == InputFileIndex)
        continue;
      printFilename(OS, *I, Format);
      OS << ":\n";
    }
  }
}
