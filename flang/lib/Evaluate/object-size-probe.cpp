//===-- lib/Evaluate/object-size-probe.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Build-time helper that measures the object size and alignment of the
// variant-backed implementation classes RealValueImpl and CharacterValue.
//
// The opaque facades RealValue (real-value.h) and CharacterValue
// (character-value.h) embed a fixed-size, suitably aligned byte buffer in place
// of their implementation so that the implementation headers do not have to be
// exposed.  The required buffer size/alignment is target-toolchain specific
// (e.g. std::string layout differs between standard libraries), so instead of
// hard-coding it we compile this tiny program with the very same toolchain and
// let it emit the matching object-sizes-generated.h header.  object-sizes.h
// then prefers that generated header (via __has_include) over its built-in
// fallback constants.
//
// When cross-compiling this program cannot be executed on the build host, so
// the fallback constants in object-sizes.h are used instead (see the CMake
// logic that guards running this probe behind NOT CMAKE_CROSSCOMPILING).
//
//===----------------------------------------------------------------------===//

#include "real-value-impl.h"
#include "flang/Evaluate/character-value-impl.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using Fortran::evaluate::value::CharacterValueImpl;
using Fortran::evaluate::value::RealValueImpl;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <object-sizes-generated.h>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::FILE *out{std::fopen(argv[1], "w")};
  if (!out) {
    std::fprintf(
        stderr, "object-size-probe: cannot open %s for writing\n", argv[1]);
    return EXIT_FAILURE;
  }

  std::fprintf(out,
      "//===-- include/flang/Evaluate/object-sizes-generated.h ---*- C++ "
      "-*-===//\n"
      "//\n"
      "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
      "Exceptions.\n"
      "// See https://llvm.org/LICENSE.txt for license information.\n"
      "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
      "//\n"
      "//"
      "===---------------------------------------------------------------------"
      "-===//\n"
      "//\n"
      "// Generated at build time by flang-evaluate-object-size-probe.\n"
      "// Do not edit; edit flang/lib/Evaluate/object-size-probe.cpp instead.\n"
      "// Included by flang/Evaluate/object-sizes.h when present on the path.\n"
      "//\n"
      "//"
      "===---------------------------------------------------------------------"
      "-===//\n"
      "\n"
      "#ifndef FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_\n"
      "#define FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_\n"
      "\n"
      "#include <cstddef>\n"
      "\n"
      "namespace Fortran::evaluate::value::detail {\n"
      "\n"
      "// Object size/alignment for RealValue / RealValueImpl.\n"
      "inline constexpr std::size_t kRealObjectSize{%zu};\n"
      "inline constexpr std::size_t kRealObjectAlign{%zu};\n"
      "\n"
      "// Object size/alignment for Character / CharacterValue.\n"
      "inline constexpr std::size_t kCharacterObjectSize{%zu};\n"
      "inline constexpr std::size_t kCharacterObjectAlign{%zu};\n"
      "\n"
      "} // namespace Fortran::evaluate::value::detail\n"
      "#endif // FORTRAN_EVALUATE_OBJECT_SIZES_GENERATED_H_\n",
      sizeof(RealValueImpl), alignof(RealValueImpl), sizeof(CharacterValueImpl),
      alignof(CharacterValueImpl));

  std::fclose(out);
  return 0;
}
