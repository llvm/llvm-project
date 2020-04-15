//===-- runtime/character.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "character.h"
#include "terminator.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime {
extern "C" {

void RTNAME(CharacterConcatenate)(Descriptor & /*temp*/,
    const Descriptor & /*operand*/, const char * /*sourceFile*/,
    int /*sourceLine*/) {
  // TODO
}

void RTNAME(CharacterConcatenateScalar)(
    Descriptor & /*temp*/, const char * /*from*/, std::size_t /*byteLength*/) {
  // TODO
}

void RTNAME(CharacterAssign)(Descriptor & /*lhs*/, const Descriptor & /*rhs*/,
    const char * /*sourceFile*/, int /*sourceLine*/) {
  // TODO
}

std::size_t RTNAME(CharacterAppend)(char *lhs, std::size_t lhsLength,
    std::size_t offset, const char *rhs, std::size_t rhsLength) {
  if (auto n{std::min(lhsLength - offset, rhsLength)}) {
    std::memcpy(lhs + offset, rhs, n);
    offset += n;
  }
  return offset;
}

void RTNAME(CharacterPad)(char *lhs, std::size_t length, std::size_t offset) {
  if (length > offset) {
    std::memset(lhs + offset, ' ', length - offset);
  }
}
}
} // namespace Fortran::runtime
