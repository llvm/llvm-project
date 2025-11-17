// UNSUPPORTED: target={{.*}}-windows-{{.*}}

// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK-NO-OUTLINE
// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-outline-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE

// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-outline-instrumentation \
// RUN:     -fsanitize-type-verify-outlined-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE
// RUN: %clang -S -fsanitize=type -emit-llvm -o - -fsanitize=type %s \
// RUN:     -fsanitize-type-verify-outlined-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE

// CHECK-LABEL: @alias
// CHECK: __tysan_app_memory_mask
// CHECK: __tysan_shadow_memory_address
// CHECK-NO-OUTLINE-NOT: call{{.*}}@__tysan_instrument_mem_inst
// CHECK-NO-OUTLINE-NOT: call{{.*}}@__tysan_instrument_with_shadow_update
// CHECK-OUTLINE: call{{.*}}@__tysan_instrument_mem_inst
// CHECK-OUTLINE: call{{.*}}@__tysan_instrument_with_shadow_update

float alias(int *ptr){
    float *aliasedPtr = (float *)ptr;
    *aliasedPtr *= 2.0f;
    return *aliasedPtr;
}
