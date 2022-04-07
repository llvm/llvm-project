// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DSTRUCT -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-STRUCT
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -USTRUCT -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTRUCT
// RUN: not %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DIMPOSSIBLE_ODD -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IMPOSSIBLE_ODD
// RUN: not %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DIMPOSSIBLE_BIG -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IMPOSSIBLE_BIG
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DPOSSIBLE_X -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-X
// RUN: not %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DIMPOSSIBLE_X -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IMPOSSIBLE_X
// RUN: not %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DIMPOSSIBLE_9BYTES -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IMPOSSIBLE_9BYTES
// RUN: not %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -DIMPOSSIBLE_9BYTES_V2 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IMPOSSIBLE_9BYTES_V2

// Make sure Clang doesn't treat |lockval| as asm input.
void _raw_spin_lock(void) {
#ifdef STRUCT
  struct {
    unsigned short owner, next;
  } lockval;
  lockval.owner = 1;
  lockval.next = 2;
#else
  int lockval;
  lockval = 3;
#endif
  asm("nop"
      : "=r"(lockval));
}
// CHECK-LABEL: _raw_spin_lock
// CHECK-LABEL: entry:

// CHECK-STRUCT:  %lockval = alloca %struct.anon, align 2
// CHECK-STRUCT:  store i16 1
// CHECK-STRUCT:  store i16 2
// CHECK-STRUCT: [[RES:%[0-9]+]] = call i32 asm "nop", "=r,~{dirflag},~{fpsr},~{flags}"()
// CHECK-STRUCT: [[CAST:%[0-9]+]] = bitcast %struct.anon* %lockval to i32*
// CHECK-STRUCT: store i32 [[RES]], i32* [[CAST]], align 2

// CHECK-NOSTRUCT: %lockval = alloca i32, align 4
// CHECK-NOSTRUCT: store i32 3
// CHECK-NOSTRUCT:  [[RES:%[0-9]+]] = call i32 asm "nop", "=r,~{dirflag},~{fpsr},~{flags}"()
// CHECK-NOSTRUCT: store i32 [[RES]], i32* %lockval, align 4

// Check Clang correctly handles a structure with padding.
void unusual_struct(void) {
  struct {
    unsigned short first;
    unsigned char second;
  } str;
  asm("nop"
      : "=r"(str));
}

// Check Clang reports an error if attempting to return a structure for which
// no direct conversion to a register is possible.
void odd_struct(void) {
#ifdef IMPOSSIBLE_ODD
  struct __attribute__((__packed__)) {
    unsigned short first;
    unsigned char second;
  } str;
  asm("nop"
      : "=r"(str));
#endif
}
// CHECK-IMPOSSIBLE_ODD: impossible constraint in asm: can't store value into a register

// Check Clang reports an error if attempting to return a big structure via a register.
void big_struct(void) {
#ifdef IMPOSSIBLE_BIG
  struct {
    long long int v1, v2, v3, v4;
  } str;
  asm("nop"
      : "=r"(str));
#endif
}
// CHECK-IMPOSSIBLE_BIG: impossible constraint in asm: can't store value into a register

// Clang is able to emit LLVM IR for an 16-byte structure.
void x_constraint_fit(void) {
#ifdef POSSIBLE_X
  struct S {
    unsigned x[4];
  } z;
  asm volatile("nop"
               : "=x"(z));
#endif
}
// CHECK-LABEL: x_constraint_fit
// CHECK-X: %z = alloca %struct.S, align 4
// CHECK-X: [[RES:%[0-9]+]] = call i128 asm sideeffect "nop", "=x,~{dirflag},~{fpsr},~{flags}"()
// CHECK-X: [[CAST:%[0-9]+]] = bitcast %struct.S* %z to i128*
// CHECK-X: store i128 [[RES]], i128* [[CAST]], align 4
// CHECK-X: ret

// Clang is unable to emit LLVM IR for a 32-byte structure.
void x_constraint_nofit(void) {
#ifdef IMPOSSIBLE_X
  struct S {
    unsigned x[8];
  } z;
  asm volatile("nop"
               : "=x"(z));
#endif
}

// CHECK-IMPOSSIBLE_X: invalid output size for constraint

// http://crbug.com/999160
// Clang used to report the following message:
//   "impossible constraint in asm: can't store struct into a register"
// for the assembly directive below, although there's no struct.
void crbug_999160_regtest(void) {
#ifdef IMPOSSIBLE_9BYTES
  char buf[9];
  asm(""
      : "=r"(buf));
#endif
}

// CHECK-IMPOSSIBLE_9BYTES: impossible constraint in asm: can't store value into a register

void crbug_999160_regtest_v2(void) {
#ifdef IMPOSSIBLE_9BYTES_V2
  char buf[9];
  asm("" : "=r"(buf) : "0"(buf));
#endif
}
// CHECK-IMPOSSIBLE_9BYTES_V2: impossible constraint in asm: can't store value into a register
