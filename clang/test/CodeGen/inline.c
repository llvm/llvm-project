// REQUIRES: x86-registered-target
//
// RUN: echo "GNU89 tests:"
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -O1 -disable-llvm-passes -emit-llvm -o - -std=gnu89 | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fexperimental-new-pass-manager -O1 -disable-llvm-passes -emit-llvm -o - -std=gnu89 | FileCheck %s --check-prefix=CHECK1
// CHECK1-LABEL: define{{.*}} i32 @foo()
// CHECK1-LABEL: define{{.*}} i32 @bar()
// CHECK1-LABEL: define{{.*}} void @unreferenced1()
// CHECK1-NOT: unreferenced2
// CHECK1-LABEL: define{{.*}} void @gnu_inline()
// CHECK1-LABEL: define{{.*}} i32 @test1
// CHECK1-LABEL: define{{.*}} i32 @test2
// CHECK1-LABEL: define{{.*}} void @test3()
// CHECK1-LABEL: define available_externally i32 @test4
// CHECK1-LABEL: define available_externally i32 @test5
// CHECK1-LABEL: define{{.*}} i32 @test6
// CHECK1-LABEL: define{{.*}} void @test7
// CHECK1: define{{.*}} i{{..}} @strlcpy
// CHECK1-NOT: test9
// CHECK1-LABEL: define{{.*}} void @testA
// CHECK1-LABEL: define{{.*}} void @testB
// CHECK1-LABEL: define{{.*}} void @testC
// CHECK1-LABEL: define available_externally i32 @ei()
// CHECK1-LABEL: define available_externally void @gnu_ei_inline()

// RUN: echo "C99 tests:"
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -O1 -disable-llvm-passes -emit-llvm -o - -std=gnu99 | FileCheck %s --check-prefix=CHECK2
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fexperimental-new-pass-manager -O1 -disable-llvm-passes -emit-llvm -o - -std=gnu99 | FileCheck %s --check-prefix=CHECK2
// CHECK2-LABEL: define{{.*}} i32 @ei()
// CHECK2-LABEL: define{{.*}} i32 @bar()
// CHECK2-NOT: unreferenced1
// CHECK2-LABEL: define{{.*}} void @unreferenced2()
// CHECK2-LABEL: define{{.*}} void @gnu_inline()
// CHECK2-LABEL: define{{.*}} i32 @test1
// CHECK2-LABEL: define{{.*}} i32 @test2
// CHECK2-LABEL: define{{.*}} void @test3
// CHECK2-LABEL: define available_externally i32 @test4
// CHECK2-LABEL: define available_externally i32 @test5
// CHECK2-LABEL: define{{.*}} i32 @test6
// CHECK2-LABEL: define{{.*}} void @test7
// CHECK2: define available_externally i{{..}} @strlcpy
// CHECK2-LABEL: define{{.*}} void @test9
// CHECK2-LABEL: define{{.*}} void @testA
// CHECK2-LABEL: define{{.*}} void @testB
// CHECK2-LABEL: define{{.*}} void @testC
// CHECK2-LABEL: define available_externally i32 @foo()
// CHECK2-LABEL: define available_externally void @gnu_ei_inline()

// RUN: echo "C++ tests:"
// RUN: %clang_cc1 -x c++ %s -triple i386-unknown-unknown -O1 -disable-llvm-passes -emit-llvm -o - -std=c++98 | FileCheck %s --check-prefix=CHECK3
// RUN: %clang_cc1 -x c++ %s -triple i386-unknown-unknown -fexperimental-new-pass-manager -O1 -disable-llvm-passes -emit-llvm -o - -std=c++98 | FileCheck %s --check-prefix=CHECK3
// CHECK3-LABEL: define{{.*}} i32 @_Z3barv()
// CHECK3-LABEL: define linkonce_odr i32 @_Z3foov()
// CHECK3-NOT: unreferenced
// CHECK3-LABEL: define available_externally void @_Z10gnu_inlinev()
// CHECK3-LABEL: define available_externally void @_Z13gnu_ei_inlinev()
// CHECK3-NOT: @_Z5testCv
// CHECK3-LABEL: define linkonce_odr i32 @_Z2eiv()

// RUN: echo "MS C Mode tests:"
// RUN: %clang_cc1 %s -triple i386-pc-win32 -O1 -disable-llvm-passes -emit-llvm -o - -std=c99 | FileCheck %s --check-prefix=CHECK4
// RUN: %clang_cc1 %s -triple i386-pc-win32 -fexperimental-new-pass-manager -O1 -disable-llvm-passes -emit-llvm -o - -std=c99 | FileCheck %s --check-prefix=CHECK4
// CHECK4-NOT: define weak_odr void @_Exit(
// CHECK4-LABEL: define weak_odr dso_local i32 @ei()
// CHECK4-LABEL: define dso_local i32 @bar()
// CHECK4-NOT: unreferenced1
// CHECK4-LABEL: define weak_odr dso_local void @unreferenced2()
// CHECK4-LABEL: define dso_local void @gnu_inline()
// CHECK4-LABEL: define linkonce_odr dso_local i32 @foo()
// CHECK4-LABEL: define available_externally dso_local void @gnu_ei_inline()

__attribute__((noreturn)) void __cdecl _exit(int _Code);
__inline void __cdecl _Exit(int status) { _exit(status); }

extern __inline int ei() { return 123; }

__inline int foo() {
  return ei();
}

int bar() { return foo(); }


__inline void unreferenced1() {}
extern __inline void unreferenced2() {}

__inline __attribute((__gnu_inline__)) void gnu_inline() {}
void (*P1)() = gnu_inline;

// PR3988
extern __inline __attribute__((gnu_inline)) void gnu_ei_inline() {}
void (*P)() = gnu_ei_inline;

// <rdar://problem/6818429>
int test1();
__inline int test1() { return 4; }
__inline int test2() { return 5; }
__inline int test2();
int test2();

void test_test1() { test1(); }
void test_test2() { test2(); }

// PR3989
extern __inline void test3() __attribute__((gnu_inline));
__inline void __attribute__((gnu_inline)) test3() {}

extern int test4(void);
extern __inline __attribute__ ((__gnu_inline__)) int test4(void)
{
  return 0;
}

void test_test4() { test4(); }

extern __inline int test5(void)  __attribute__ ((__gnu_inline__));
extern __inline int __attribute__ ((__gnu_inline__)) test5(void)
{
  return 0;
}

void test_test5() { test5(); }

// PR10233

__inline int test6() { return 0; }
extern int test6();


// No PR#, but this once crashed clang in C99 mode due to buggy extern inline
// redeclaration detection.
void test7() { }
void test7();

// PR11062; the fact that the function is named strlcpy matters here.
inline __typeof(sizeof(int)) strlcpy(char *dest, const char *src, __typeof(sizeof(int)) size) { return 3; }
void test8() { strlcpy(0,0,0); }

// PR10657; the test crashed in C99 mode
extern inline void test9() { }
void test9();

inline void testA() {}
void testA();

void testB();
inline void testB() {}
extern void testB();

extern inline void testC() {}
inline void testC();
