// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions %s -O2 -o - | FileCheck %s --check-prefix=X86-64
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions %s -O2 -o - | FileCheck %s --check-prefix=X86-32
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions %s -O2 -o - | FileCheck %s --check-prefix=ARM-DARWIN
// RUN: %clang_cc1 -triple arm-unknown-gnueabi -emit-llvm -fcxx-exceptions -fexceptions %s -O2 -o - | FileCheck %s --check-prefix=ARM-EABI
// RUN: %clang_cc1 -triple mipsel-unknown-unknown -emit-llvm -fcxx-exceptions -fexceptions %s -O2 -o - | FileCheck %s --check-prefix=MIPS

void foo();
void test() {
  try {
    foo();
  } catch (int *&i) {
    *i = 5;
  }
}

// PR10789: different platforms have different sizes for struct UnwindException.

// X86-64:          [[T0:%.*]] = tail call i8* @__cxa_begin_catch(i8* [[EXN:%.*]]) nounwind
// X86-64-NEXT:     [[T1:%.*]] = getelementptr i8* [[EXN]], i64 32
// X86-32:          [[T0:%.*]] = tail call i8* @__cxa_begin_catch(i8* [[EXN:%.*]]) nounwind
// X86-32-NEXT:     [[T1:%.*]] = getelementptr i8* [[EXN]], i64 32
// ARM-DARWIN:      [[T0:%.*]] = tail call i8* @__cxa_begin_catch(i8* [[EXN:%.*]]) nounwind
// ARM-DARWIN-NEXT: [[T1:%.*]] = getelementptr i8* [[EXN]], i64 32
// ARM-EABI:        [[T0:%.*]] = tail call i8* @__cxa_begin_catch(i8* [[EXN:%.*]]) nounwind
// ARM-EABI-NEXT:   [[T1:%.*]] = getelementptr i8* [[EXN]], i32 88
// MIPS:            [[T0:%.*]] = tail call i8* @__cxa_begin_catch(i8* [[EXN:%.*]]) nounwind
// MIPS-NEXT:       [[T1:%.*]] = getelementptr i8* [[EXN]], i32 24

