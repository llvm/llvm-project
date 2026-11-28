// Test that compiler-generated helper functions carry the "use-sample-profile"
// attribute when sample profile is enabled, so that they are inline-compatible
// with the functions that call them.

// RUN: printf 'ReadNode:100000:100000\n 1: 100000\n' > %t.prof

// Itanium thread-local wrapper.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++14 -O2 -disable-llvm-passes \
// RUN:   -fprofile-sample-use=%t.prof -DTLS_WRAPPER %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=TLS

// MSVC default/copy constructor closures.
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -std=c++14 -O2 -disable-llvm-passes \
// RUN:   -fcxx-exceptions -fms-extensions -fprofile-sample-use=%t.prof \
// RUN:   -DMSVC_CLOSURE %s -emit-llvm -o - | FileCheck %s --check-prefix=MSVC

// Incremental-extensions top-level statement function.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fincremental-extensions \
// RUN:   -O2 -disable-llvm-passes -fprofile-sample-use=%t.prof -DTOP_LEVEL %s \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=STMT

#ifdef TLS_WRAPPER
extern thread_local int tls;
extern "C" int ReadNode() { return tls; }
#endif

#ifdef MSVC_CLOSURE
struct DefaultClosure {
  __declspec(dllexport) DefaultClosure(int n = 42) {}
};
struct CopyClosure {
  CopyClosure(const CopyClosure &, int n = 42);
};
void trigger(CopyClosure &c) { throw c; }
#endif

#ifdef TOP_LEVEL
static int helper() { return 42; }
int n;
n = helper();
#endif

// TLS: define{{.*}} @ReadNode{{.*}} #[[READNODE_ATTR:[0-9]+]]
// TLS: define{{.*}} @_ZTW3tls{{.*}} #[[WRAPPER_ATTR:[0-9]+]]
// TLS: attributes #[[READNODE_ATTR]] = {{.*}} "use-sample-profile"
// TLS: attributes #[[WRAPPER_ATTR]] = {{.*}} "use-sample-profile"

// MSVC: define{{.*}}@"??_FDefaultClosure@@QAEXXZ"{{.*}} #[[DEFAULT_ATTR:[0-9]+]]
// MSVC: define{{.*}}@"??_OCopyClosure@@QAEXABU0@@Z"{{.*}} #[[COPY_ATTR:[0-9]+]]
// MSVC: attributes #[[DEFAULT_ATTR]] = {{.*}} "use-sample-profile"
// MSVC: attributes #[[COPY_ATTR]] = {{.*}} "use-sample-profile"

// STMT: define internal void @__stmts__0{{.*}} #[[STMT_ATTR:[0-9]+]]
// STMT: attributes #[[STMT_ATTR]] = {{.*}} "use-sample-profile"
