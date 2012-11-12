// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

void foo(const unsigned int) {}
// CHECK: "\01?foo@@YAXI@Z"

void foo(const double) {}
// CHECK: "\01?foo@@YAXN@Z"

void bar(const volatile double) {}
// CHECK: "\01?bar@@YAXN@Z"

void foo_pad(char * x) {}
// CHECK: "\01?foo_pad@@YAXPAD@Z"

void foo_pbd(const char * x) {}
// CHECK: "\01?foo_pbd@@YAXPBD@Z"

void foo_pcd(volatile char * x) {}
// CHECK: "\01?foo_pcd@@YAXPCD@Z"

void foo_qad(char * const x) {}
// CHECK: "\01?foo_qad@@YAXQAD@Z"

void foo_rad(char * volatile x) {}
// CHECK: "\01?foo_rad@@YAXRAD@Z"

void foo_sad(char * const volatile x) {}
// CHECK: "\01?foo_sad@@YAXSAD@Z"

void foo_papad(char ** x) {}
// CHECK: "\01?foo_papad@@YAXPAPAD@Z"

void foo_papbd(char const ** x) {}
// CHECK: "\01?foo_papbd@@YAXPAPBD@Z"

void foo_papcd(char volatile ** x) {}
// CHECK: "\01?foo_papcd@@YAXPAPCD@Z"

void foo_pbqad(char * const* x) {}
// CHECK: "\01?foo_pbqad@@YAXPBQAD@Z"

void foo_pcrad(char * volatile* x) {}
// CHECK: "\01?foo_pcrad@@YAXPCRAD@Z"

void foo_qapad(char ** const x) {}
// CHECK: "\01?foo_qapad@@YAXQAPAD@Z"

void foo_rapad(char ** volatile x) {}
// CHECK: "\01?foo_rapad@@YAXRAPAD@Z"

void foo_pbqbd(const char * const* x) {}
// CHECK: "\01?foo_pbqbd@@YAXPBQBD@Z"

void foo_pbqcd(volatile char * const* x) {}
// CHECK: "\01?foo_pbqcd@@YAXPBQCD@Z"

void foo_pcrbd(const char * volatile* x) {}
// CHECK: "\01?foo_pcrbd@@YAXPCRBD@Z"

void foo_pcrcd(volatile char * volatile* x) {}
// CHECK: "\01?foo_pcrcd@@YAXPCRCD@Z"

typedef double Vector[3];

void foo(Vector*) {}
// CHECK: "\01?foo@@YAXPAY02N@Z"

void foo(Vector) {}
// CHECK: "\01?foo@@YAXQAN@Z"

void foo_const(const Vector) {}
// CHECK: "\01?foo_const@@YAXQBN@Z"

void foo_volatile(volatile Vector) {}
// CHECK: "\01?foo_volatile@@YAXQCN@Z"

void foo(Vector*, const Vector, const double) {}
// CHECK: "\01?foo@@YAXPAY02NQBNN@Z"
