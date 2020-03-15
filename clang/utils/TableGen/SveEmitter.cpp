//===- SveEmitter.cpp - Generate arm_sve.h for use with clang -*- C++ -*-===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_sve.h, which includes
// a declaration and definition of each function specified by the ARM C/C++
// Language Extensions (ACLE).
//
// For details, visit:
//  https://developer.arm.com/architectures/system-architectures/software-standards/acle
//
// Each SVE instruction is implemented in terms of 1 or more functions which
// are suffixed with the element type of the input vectors.  Functions may be
// implemented in terms of generic vector operations such as +, *, -, etc. or
// by calling a __builtin_-prefixed function which will be handled by clang's
// CodeGen library.
//
// See also the documentation in include/clang/Basic/arm_sve.td.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/Error.h"
#include <string>
#include <sstream>
#include <set>
#include <cctype>

using namespace llvm;

//===----------------------------------------------------------------------===//
// SVEEmitter
//===----------------------------------------------------------------------===//

namespace {

class SVEEmitter {
private:
  RecordKeeper &Records;

public:
  SVEEmitter(RecordKeeper &R) : Records(R) {}

  // run - Emit arm_sve.h
  void run(raw_ostream &o);
};

} // end anonymous namespace


//===----------------------------------------------------------------------===//
// SVEEmitter implementation
//===----------------------------------------------------------------------===//

void SVEEmitter::run(raw_ostream &OS) {
  OS << "/*===---- arm_sve.h - ARM SVE intrinsics "
        "-----------------------------------===\n"
        " *\n"
        " *\n"
        " * Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n"
        " * See https://llvm.org/LICENSE.txt for license information.\n"
        " * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        " *\n"
        " *===-----------------------------------------------------------------"
        "------===\n"
        " */\n\n";

  OS << "#ifndef __ARM_SVE_H\n";
  OS << "#define __ARM_SVE_H\n\n";

  OS << "#if !defined(__ARM_FEATURE_SVE)\n";
  OS << "#error \"SVE support not enabled\"\n";
  OS << "#else\n\n";

  OS << "#include <stdint.h>\n\n";
  OS << "#ifndef  __cplusplus\n";
  OS << "#include <stdbool.h>\n";
  OS << "#endif\n\n";

  OS << "typedef __fp16 float16_t;\n";
  OS << "typedef float float32_t;\n";
  OS << "typedef double float64_t;\n";
  OS << "typedef bool bool_t;\n\n";

  OS << "typedef __SVInt8_t svint8_t;\n";
  OS << "typedef __SVInt16_t svint16_t;\n";
  OS << "typedef __SVInt32_t svint32_t;\n";
  OS << "typedef __SVInt64_t svint64_t;\n";
  OS << "typedef __SVUint8_t svuint8_t;\n";
  OS << "typedef __SVUint16_t svuint16_t;\n";
  OS << "typedef __SVUint32_t svuint32_t;\n";
  OS << "typedef __SVUint64_t svuint64_t;\n";
  OS << "typedef __SVFloat16_t svfloat16_t;\n";
  OS << "typedef __SVFloat32_t svfloat32_t;\n";
  OS << "typedef __SVFloat64_t svfloat64_t;\n";
  OS << "typedef __SVBool_t  svbool_t;\n\n";

  OS << "#define svld1_u8(...) __builtin_sve_svld1_u8(__VA_ARGS__)\n";
  OS << "#define svld1_u16(...) __builtin_sve_svld1_u16(__VA_ARGS__)\n";
  OS << "#define svld1_u32(...) __builtin_sve_svld1_u32(__VA_ARGS__)\n";
  OS << "#define svld1_u64(...) __builtin_sve_svld1_u64(__VA_ARGS__)\n";
  OS << "#define svld1_s8(...) __builtin_sve_svld1_s8(__VA_ARGS__)\n";
  OS << "#define svld1_s16(...) __builtin_sve_svld1_s16(__VA_ARGS__)\n";
  OS << "#define svld1_s32(...) __builtin_sve_svld1_s32(__VA_ARGS__)\n";
  OS << "#define svld1_s64(...) __builtin_sve_svld1_s64(__VA_ARGS__)\n";
  OS << "#define svld1_f16(...) __builtin_sve_svld1_f16(__VA_ARGS__)\n";
  OS << "#define svld1_f32(...) __builtin_sve_svld1_f32(__VA_ARGS__)\n";
  OS << "#define svld1_f64(...) __builtin_sve_svld1_f64(__VA_ARGS__)\n";

  OS << "#endif /*__ARM_FEATURE_SVE */\n";
  OS << "#endif /* __ARM_SVE_H */\n";
}

namespace clang {
void EmitSveHeader(RecordKeeper &Records, raw_ostream &OS) {
  SVEEmitter(Records).run(OS);
}

} // End namespace clang
