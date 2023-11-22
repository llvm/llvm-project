// RUN: %clang --target=s390x-linux -S -emit-llvm -o - %s | FileCheck %s
//
// Test that a global variable with an incomplete type gets the minimum
// alignment of 2 per the ABI if no alignment was specified by user.
//
// CHECK:      @VarNoAl {{.*}} align 2
// CHECK-NEXT: @VarExplAl1  {{.*}} align 1
// CHECK-NEXT: @VarExplAl4  {{.*}} align 4

// No alignemnt specified by user.
struct incomplete_ty_noal;
extern struct incomplete_ty_noal VarNoAl;
struct incomplete_ty_noal *fun0 (void)
{
  return &VarNoAl;
}

// User-specified alignment of 1.
struct incomplete_ty_al1;
extern struct incomplete_ty_al1 __attribute__((aligned(1))) VarExplAl1;
struct incomplete_ty_al1 *fun1 (void)
{
  return &VarExplAl1;
}

// User-specified alignment of 4.
struct incomplete_ty_al4;
extern struct incomplete_ty_al4 __attribute__((aligned(4))) VarExplAl4;
struct incomplete_ty_al4 *fun2 (void)
{
  return &VarExplAl4;
}
