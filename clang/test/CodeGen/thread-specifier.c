// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @b = external thread_local global
// CHECK: @d.e = internal thread_local global
// CHECK: @d.f = internal thread_local global
// CHECK: @f.a = internal thread_local(initialexec) global
// CHECK: @a = thread_local global
// CHECK: @g = thread_local global
// CHECK: @h = thread_local(localdynamic) global
// CHECK: @i = thread_local(initialexec) global
// CHECK: @j = thread_local(localexec) global

__thread int a;
extern __thread int b;
int c() { return *&b; }
int d() {
  __thread static int e;
  __thread static union {float a; int b;} f = {.b = 1};
  return 0;
}

__thread int g __attribute__((tls_model("global-dynamic")));
__thread int h __attribute__((tls_model("local-dynamic")));
__thread int i __attribute__((tls_model("initial-exec")));
__thread int j __attribute__((tls_model("local-exec")));

int f() {
  __thread static int a __attribute__((tls_model("initial-exec")));
  return a++;
}
