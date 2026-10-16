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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>

using Fortran::evaluate::value::CharacterValueImpl;
using Fortran::evaluate::value::IntegerValueImpl;
using Fortran::evaluate::value::RealValueImpl;
using namespace llvm;

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
    cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool> WriteIfChanged(
    "write-if-changed", cl::desc("Only write output if it changed"));

static int reportError(const char *ProgName, Twine Msg) {
  errs() << ProgName << ": " << Msg;
  errs().flush();
  return 1;
}

static int WriteOutput(
    const char *argv0, StringRef Filename, StringRef Content) {
  if (WriteIfChanged) {
    // Only updates the real output file if there are any differences.
    // This prevents recompilation of all the files depending on it if there
    // aren't any.
    if (auto ExistingOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/true))
      if (std::move(ExistingOrErr.get())->getBuffer() == Content)
        return 0;
  }
  std::error_code EC;
  ToolOutputFile OutFile(Filename, EC, sys::fs::OF_Text);
  if (EC)
    return reportError(
        argv0, "error opening " + Filename + ": " + EC.message() + "\n");
  OutFile.os() << Content;
  OutFile.keep();

  return 0;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  SmallString<1024> Buffer;
  raw_svector_ostream OS(Buffer);

  OS << llvm::format(
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

  WriteOutput("object-size-probe", OutputFilename, OS.str());

  return EXIT_SUCCESS;
}
