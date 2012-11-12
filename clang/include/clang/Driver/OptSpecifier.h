//===--- OptSpecifier.h - Option Specifiers ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OPTSPECIFIER_H
#define CLANG_DRIVER_OPTSPECIFIER_H

namespace clang {
namespace driver {
  class Option;

  /// OptSpecifier - Wrapper class for abstracting references to option IDs.
  class OptSpecifier {
    unsigned ID;

  private:
    explicit OptSpecifier(bool); // DO NOT IMPLEMENT

  public:
    OptSpecifier() : ID(0) {}
    /*implicit*/ OptSpecifier(unsigned _ID) : ID(_ID) {}
    /*implicit*/ OptSpecifier(const Option *Opt);

    bool isValid() const { return ID != 0; }

    unsigned getID() const { return ID; }

    bool operator==(OptSpecifier Opt) const { return ID == Opt.getID(); }
    bool operator!=(OptSpecifier Opt) const { return !(*this == Opt); }
  };
}
}

#endif
