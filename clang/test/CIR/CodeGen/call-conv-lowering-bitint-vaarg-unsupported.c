// RUN: not %clang_cc1 -triple i386-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=I386
// RUN: not %clang_cc1 -triple powerpc64-unknown-linux-gnu -fclangir \
// RUN:   -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PPC64
// RUN: not %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=AARCH64

typedef signed _BitInt(31) value_t;

value_t fetch(__builtin_va_list ap) {
  return __builtin_va_arg(ap, value_t);
}

// I386: error: 'cir.va_arg' op i386 va_arg lowering for _BitInt not yet implemented in CallConvLowering
// PPC64: error: 'cir.va_arg' op PowerPC64 va_arg lowering for _BitInt not yet implemented in CallConvLowering
// AARCH64: error: 'cir.va_arg' op AArch64 va_arg lowering for _BitInt not yet implemented in CallConvLowering
