
//===--- CommandLineSourceLoc.h - Parsing for source locations-*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Command line parsing for source locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMMANDLINESOURCELOC_H
#define LLVM_CLANG_FRONTEND_COMMANDLINESOURCELOC_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

/// \brief A source location that has been parsed on the command line.
struct ParsedSourceLocation {
  std::string FileName;
  unsigned Line;
  unsigned Column;

public:
  /// Construct a parsed source location from a string; the Filename is empty on
  /// error.
  static ParsedSourceLocation FromString(StringRef Str) {
    ParsedSourceLocation PSL;
    std::pair<StringRef, StringRef> ColSplit = Str.rsplit(':');
    std::pair<StringRef, StringRef> LineSplit =
      ColSplit.first.rsplit(':');

    // If both tail splits were valid integers, return success.
    if (!ColSplit.second.getAsInteger(10, PSL.Column) &&
        !LineSplit.second.getAsInteger(10, PSL.Line)) {
      PSL.FileName = LineSplit.first;

      // On the command-line, stdin may be specified via "-". Inside the
      // compiler, stdin is called "<stdin>".
      if (PSL.FileName == "-")
        PSL.FileName = "<stdin>";
    }

    return PSL;
  }
};

}

namespace llvm {
  namespace cl {
    /// \brief Command-line option parser that parses source locations.
    ///
    /// Source locations are of the form filename:line:column.
    template<>
    class parser<clang::ParsedSourceLocation> final
      : public basic_parser<clang::ParsedSourceLocation> {
    public:
      inline bool parse(Option &O, StringRef ArgName, StringRef ArgValue,
                 clang::ParsedSourceLocation &Val);
    };

    bool
    parser<clang::ParsedSourceLocation>::
    parse(Option &O, StringRef ArgName, StringRef ArgValue,
          clang::ParsedSourceLocation &Val) {
      using namespace clang;

      Val = ParsedSourceLocation::FromString(ArgValue);
      if (Val.FileName.empty()) {
        errs() << "error: "
               << "source location must be of the form filename:line:column\n";
        return true;
      }

      return false;
    }
  }
}

#endif
