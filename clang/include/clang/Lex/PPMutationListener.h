//===--- PPMutationListener.h - Preprocessor Mutation Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PPMutationListener interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LEX_PPTMUTATIONLISTENER_H
#define LLVM_CLANG_LEX_PPTMUTATIONLISTENER_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class MacroInfo;

/// \brief A record that describes an update to a macro that was
/// originally loaded to an AST file and has been modified within the
/// current translation unit.
struct MacroUpdate {
  /// \brief The source location at which this macro was #undef'd.
  SourceLocation UndefLoc;
};

/// \brief An abstract interface that should be implemented by
/// listeners that want to be notified when a preprocessor entity gets
/// modified after its initial creation.
class PPMutationListener {
public:
  virtual ~PPMutationListener();

  /// \brief A macro has been #undef'd.
  virtual void UndefinedMacro(MacroInfo *MI) { }
};

} // end namespace clang

#endif
