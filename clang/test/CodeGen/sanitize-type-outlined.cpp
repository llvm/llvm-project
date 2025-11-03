// UNSUPPORTED: target={{.*}}-windows-{{.*}}

// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK-NO-OUTLINE
// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-outline-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE

// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-outline-instrumentation \
// RUN:     -fsanitize-type-verify-outlined-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE-VERIFY
// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-verify-outlined-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-VERIFY

// CHECK-NO-OUTLINE-NOT: call{{.*}}@__tysan_instrument_mem_inst
// CHECK-OUTLINE: call{{.*}}@__tysan_instrument_mem_inst
// CHECK-OUTLINE-VERIFY: call{{.*}}@__tysan_instrument_mem_inst
// CHECK-VERIFY: call{{.*}}@__tysan_instrument_mem_inst

float alias(int *ptr){
    float *aliasedPtr = (float *)ptr;
    return *aliasedPtr;
}
