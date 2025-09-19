// RUN: %clang_cc1 -ast-dump %s | FileCheck %s


// CHECK: FunctionDecl [[PTR:0x[a-z0-9]*]] {{.*}}func 'void () __attribute__((cfi_unchecked_callee))'
__attribute__((cfi_unchecked_callee))
void func(void);

// CHECK-NEXT: FunctionDecl {{0x[a-z0-9]*}} prev [[PTR]] {{.*}}func 'void () __attribute__((cfi_unchecked_callee))'
void func(void) {}
