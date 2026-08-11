//===-- tools/object-size-probe/object-size-probe.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Automatic deduction of the opaque object size/alignment used by the
// IntegerValue, RealValue, and CharacterValue facades (integer-value.h,
// real-value.h, character-value.h).
//
// These are similar to the pImpl-idiom, except that instead of the facade
// storing a pointer to the implementation-object (IntegerValueImpl,
// RealValueImpl, CharacterValueImpl), it is reinterpret-casted over the facade
// object. This requires both to have the same object sizes. A `sizeof(*Impl)`
// would defeat the purpose of hiding the implementation. Instead, we probe the
// object size at build time.
//
// This program is compiled and executed to generate a header file containing
// sizes of the implementation objects.
//
//===----------------------------------------------------------------------===//

#define FLANG_OBJECT_SIZE_PROBE

#include "character-value-impl.h"
#include "integer-value-impl.h"
#include "real-value-impl.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>

using Fortran::evaluate::value::CharacterValueImpl;
using Fortran::evaluate::value::IntegerValueImpl;
using Fortran::evaluate::value::RealValueImpl;

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: " << argv[0] << " <object-sizes-generated.h>\n";
    return EXIT_FAILURE;
  }

  std::error_code ec;
  llvm::ToolOutputFile out(argv[1], ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "object-size-probe: cannot open " << argv[1]
                 << " for writing: " << ec.message() << '\n';
    return EXIT_FAILURE;
  }

  out.os() << llvm::format(
      R"(
//===-- object-sizes-generated.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generated at build time by flang-object-size-probe.
// Do not edit; edit flang/tools/object-size-probe/object-size-probe.cpp instead.
// Included by flang/Evaluate/object-sizes.h when present on the path.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_
#define FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_

#include <cstddef>

namespace Fortran::evaluate::value::detail {

// Object size/alignment for IntegerValue / IntegerValueImpl
inline constexpr std::size_t kIntegerObjectSize{%zu};
inline constexpr std::size_t kIntegerObjectAlign{%zu};

// Object size/alignment for RealValue / RealValueImpl
inline constexpr std::size_t kRealObjectSize{%zu};
inline constexpr std::size_t kRealObjectAlign{%zu};

// Object size/alignment for CharacterValue / CharacterValueImpl
inline constexpr std::size_t kCharacterObjectSize{%zu};
inline constexpr std::size_t kCharacterObjectAlign{%zu};

} // namespace Fortran::evaluate::value::detail
#endif // FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_
)",
      sizeof(IntegerValueImpl), alignof(IntegerValueImpl),
      sizeof(RealValueImpl), alignof(RealValueImpl), sizeof(CharacterValueImpl),
      alignof(CharacterValueImpl));

  out.keep();
  return EXIT_SUCCESS;
}
