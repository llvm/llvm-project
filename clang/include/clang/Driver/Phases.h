//===--- Phases.h - Transformations on Driver Types -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_PHASES_H_
#define CLANG_DRIVER_PHASES_H_

namespace clang {
namespace driver {
namespace phases {
  /// ID - Ordered values for successive stages in the
  /// compilation process which interact with user options.
  enum ID {
    Preprocess,
    Precompile,
    Compile,
    Assemble,
    Link
  };

  const char *getPhaseName(ID Id);

} // end namespace phases
} // end namespace driver
} // end namespace clang

#endif
