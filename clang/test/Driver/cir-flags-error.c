// REQUIRES: cir-support

// RUN: not %clang -emit-cir %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=EMIT_CIR_ERROR
// EMIT_CIR_ERROR: error: -emit-cir is only valid with -fclangir

// RUN: %clang -fclangir -emit-cir %s -o /dev/null 2>&1 | FileCheck %s --allow-empty -check-prefix=EMIT_CIR_OK
// EMIT_CIR_OK-NOT: error: -emit-cir is only valid with -fclangir

int main(void) {
  return 0;
}
