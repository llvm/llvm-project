//===--- InputInfo.h - Input Source & Type Information ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_INPUTINFO_H
#define LLVM_CLANG_LIB_DRIVER_INPUTINFO_H

#include "clang/Driver/Types.h"
#include "llvm/Option/Arg.h"
#include <cassert>
#include <string>

namespace clang {
namespace driver {

/// InputInfo - Wrapper for information about an input source.
class InputInfo {
  // FIXME: The distinction between filenames and inputarg here is
  // gross; we should probably drop the idea of a "linker
  // input". Doing so means tweaking pipelining to still create link
  // steps when it sees linker inputs (but not treat them as
  // arguments), and making sure that arguments get rendered
  // correctly.
  enum Class {
    Nothing,
    Filename,
    InputArg,
    Pipe
  };

  union {
    const char *Filename;
    const llvm::opt::Arg *InputArg;
  } Data;
  Class Kind;
  types::ID Type;
  const char *BaseInput;

public:
  InputInfo() {}
  InputInfo(types::ID _Type, const char *_BaseInput)
    : Kind(Nothing), Type(_Type), BaseInput(_BaseInput) {
  }
  InputInfo(const char *_Filename, types::ID _Type, const char *_BaseInput)
    : Kind(Filename), Type(_Type), BaseInput(_BaseInput) {
    Data.Filename = _Filename;
  }
  InputInfo(const llvm::opt::Arg *_InputArg, types::ID _Type,
            const char *_BaseInput)
      : Kind(InputArg), Type(_Type), BaseInput(_BaseInput) {
    Data.InputArg = _InputArg;
  }

  bool isNothing() const { return Kind == Nothing; }
  bool isFilename() const { return Kind == Filename; }
  bool isInputArg() const { return Kind == InputArg; }
  types::ID getType() const { return Type; }
  const char *getBaseInput() const { return BaseInput; }

  const char *getFilename() const {
    assert(isFilename() && "Invalid accessor.");
    return Data.Filename;
  }
  const llvm::opt::Arg &getInputArg() const {
    assert(isInputArg() && "Invalid accessor.");
    return *Data.InputArg;
  }

  /// getAsString - Return a string name for this input, for
  /// debugging.
  std::string getAsString() const {
    if (isFilename())
      return std::string("\"") + getFilename() + '"';
    else if (isInputArg())
      return "(input arg)";
    else
      return "(nothing)";
  }
};

} // end namespace driver
} // end namespace clang

#endif
