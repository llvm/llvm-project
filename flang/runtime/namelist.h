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

#include "non-tbp-dio.h"

#include <cstddef>

namespace Fortran::runtime {
class Descriptor;
class IoStatementState;
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
  const char *groupName{nullptr}; // NUL-terminated lower-case
  std::size_t items{0};
  const Item *item{nullptr}; // in original declaration order

  // When the uses of a namelist group appear in scopes with distinct sets
  // of non-type-bound defined formatted I/O interfaces, they require the
  // use of distinct NamelistGroups pointing to distinct NonTbpDefinedIoTables.
  // Multiple NamelistGroup instances may share a NonTbpDefinedIoTable..
  const NonTbpDefinedIoTable *nonTbpDefinedIo{nullptr};
};

// Look ahead on input for a '/' or an identifier followed by a '=', '(', or '%'
// character; for use in disambiguating a name-like value (e.g. F or T) from a
// NAMELIST group item name and for coping with short arrays.  Always false
// when not reading a NAMELIST.
bool IsNamelistNameOrSlash(IoStatementState &);

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_NAMELIST_H_
