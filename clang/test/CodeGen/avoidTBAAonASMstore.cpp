// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -O2 -disable-llvm-passes -fasm-blocks %s -emit-llvm -o - | FileCheck --check-prefix=STORE-LINE %s
double foo(double z) {
  // STORE-LINE-LABEL: define{{.*}} double @_Z3food
  unsigned short ControlWord;
  __asm { fnstcw word ptr[ControlWord]}
  ;
  // STORE-LINE: store i64 %{{.*}}, ptr %{{.*}},
  // STORE-LINE-NOT: align 4, !tbaa
  // STORE-LINE-SAME: align 4
  return z;
}
