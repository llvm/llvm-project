// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv67t -triple hexagon-unknown-elf \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv67t -triple hexagon-unknown-elf -x c++ \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// RUN: not %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -x c++ \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --implicit-check-not='error:' \
// RUN:   --check-prefix=CHECK-ERR-CXX %s

// RUN: not %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -std=c99 \
// RUN:   -Wimplicit-function-declaration -Werror -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --implicit-check-not='error:' --check-prefix=CHECK-ERR-C99 %s

#include <hexagon_protos.h>

void test_audio() {
  unsigned int b;
  unsigned long long c;

  // CHECK-ERR-CXX: error: use of undeclared identifier 'Q6_R_clip_RI'
  // CHECK-ERR-C99: error: call to undeclared function 'Q6_R_clip_RI'
  // CHECK: call i32 @llvm.hexagon.A7.clip
  b = Q6_R_clip_RI(b, 9);

  // CHECK-ERR-CXX: error: use of undeclared identifier 'Q6_P_cround_PI'
  // CHECK-ERR-C99: error: call to undeclared function 'Q6_P_cround_PI'
  // CHECK: call i64 @llvm.hexagon.A7.cround
  c = Q6_P_cround_PI(c, 12);
}
