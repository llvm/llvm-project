//===----- EditedSource.h - Collection of source edits ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EDIT_EDITSRECEIVER_H
#define LLVM_CLANG_EDIT_EDITSRECEIVER_H

#include "clang/Basic/LLVM.h"

namespace clang {
  class SourceLocation;
  class CharSourceRange;

namespace edit {

class EditsReceiver {
public:
  virtual ~EditsReceiver() { }

  virtual void insert(SourceLocation loc, StringRef text) = 0;
  virtual void replace(CharSourceRange range, StringRef text) = 0;
  /// \brief By default it calls replace with an empty string.
  virtual void remove(CharSourceRange range);
};

}

} // end namespace clang

#endif
