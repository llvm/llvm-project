//===-- runtime/namelist.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the data structure used for NAMELIST I/O

#ifndef FORTRAN_RUNTIME_NAMELIST_H_
#define FORTRAN_RUNTIME_NAMELIST_H_

#include <cstddef>

namespace Fortran::runtime {
class Descriptor;
} // namespace Fortran::runtime

namespace Fortran::runtime::io {

// A NAMELIST group is a named ordered collection of distinct variable names.
// It is packaged by lowering into an instance of this class.
// If all the items are variables with fixed addresses, the NAMELIST group
// description can be in a read-only section.
class NamelistGroup {
public:
  struct Item {
    const char *name; // NUL-terminated lower-case
    const Descriptor &descriptor;
  };
  const char *groupName; // NUL-terminated lower-case
  std::size_t items;
  const Item *item; // in original declaration order
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_NAMELIST_H_
