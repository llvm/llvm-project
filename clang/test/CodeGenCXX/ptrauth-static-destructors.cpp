// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:  | FileCheck %s --check-prefix=CXAATEXIT

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:    -fno-use-cxa-atexit | FileCheck %s --check-prefixes=ATEXIT,DARWIN

// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:  | FileCheck %s --check-prefix=CXAATEXIT

// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:    -fno-use-cxa-atexit | FileCheck %s --check-prefixes=ATEXIT,ELF

class Foo {
 public:
  ~Foo() {
  }
};

Foo global;

// CXAATEXIT: define internal void @__cxx_global_var_init()
// CXAATEXIT:   call i32 @__cxa_atexit(ptr ptrauth (ptr @_ZN3FooD1Ev, i32 0), ptr @global, ptr @__dso_handle)


// ATEXIT: define internal void @__cxx_global_var_init()
// ATEXIT:   %{{.*}} = call i32 @atexit(ptr ptrauth (ptr @__dtor_global, i32 0))

// DARWIN: define internal void @__dtor_global() {{.*}} section "__TEXT,__StaticInit,regular,pure_instructions" {
// ELF:    define internal void @__dtor_global() {{.*}} section ".text.startup" {
// DARWIN:   %{{.*}} = call ptr @_ZN3FooD1Ev(ptr @global)
// ELF:      call void @_ZN3FooD1Ev(ptr @global)
