//===-- include/flang/Semantics/attr.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_ATTR_H_
#define FORTRAN_SEMANTICS_ATTR_H_

#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include <cinttypes>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {

// All available attributes.
ENUM_CLASS(Attr, ABSTRACT, ALLOCATABLE, ASYNCHRONOUS, BIND_C, CONTIGUOUS,
    DEFERRED, ELEMENTAL, EXTENDS, EXTERNAL, IMPURE, INTENT_IN, INTENT_INOUT,
    INTENT_OUT, INTRINSIC, MODULE, NON_OVERRIDABLE, NON_RECURSIVE, NOPASS,
    OPTIONAL, PARAMETER, PASS, POINTER, PRIVATE, PROTECTED, PUBLIC, PURE,
    RECURSIVE, SAVE, TARGET, VALUE, VOLATILE)

// Set of attributes
class Attrs : public common::EnumSet<Attr, Attr_enumSize> {
private:
  using enumSetType = common::EnumSet<Attr, Attr_enumSize>;

public:
  using enumSetType::enumSetType;
  Attrs(const enumSetType &attrs) : enumSetType(attrs) {}
  Attrs(enumSetType &&attrs) : enumSetType(std::move(attrs)) {}
  constexpr bool HasAny(const Attrs &x) const { return !(*this & x).none(); }
  constexpr bool HasAll(const Attrs &x) const { return (~*this & x).none(); }
  // Internal error if any of these attributes are not in allowed.
  void CheckValid(const Attrs &allowed) const;

private:
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Attrs &);
};

// Return string representation of attr that matches Fortran source.
std::string AttrToString(Attr attr);

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, Attr attr);
llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const Attrs &attrs);
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_ATTR_H_
