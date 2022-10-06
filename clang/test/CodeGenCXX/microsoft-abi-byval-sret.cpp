// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i686-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck %s

struct A {
  A() : a(42) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
};

struct B {
  A foo(A o);
  A __cdecl bar(A o);
  A __stdcall baz(A o);
  A __fastcall qux(A o);
};

A B::foo(A x) {
  return x;
}

// CHECK-LABEL: define dso_local x86_thiscallcc ptr @"?foo@B@@QAE?AUA@@U2@@Z"
// CHECK:       (ptr noundef %this, ptr inalloca(<{ ptr, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ ptr, %struct.A }>, ptr %{{.*}}, i32 0, i32 0
// CHECK:   load ptr, ptr
// CHECK:   ret ptr

A B::bar(A x) {
  return x;
}

// CHECK-LABEL: define dso_local ptr @"?bar@B@@QAA?AUA@@U2@@Z"
// CHECK:       (ptr inalloca(<{ ptr, ptr, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ ptr, ptr, %struct.A }>, ptr %{{.*}}, i32 0, i32 1
// CHECK:   load ptr, ptr
// CHECK:   ret ptr

A B::baz(A x) {
  return x;
}

// CHECK-LABEL: define dso_local x86_stdcallcc ptr @"?baz@B@@QAG?AUA@@U2@@Z"
// CHECK:       (ptr inalloca(<{ ptr, ptr, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ ptr, ptr, %struct.A }>, ptr %{{.*}}, i32 0, i32 1
// CHECK:   load ptr, ptr
// CHECK:   ret ptr

A B::qux(A x) {
  return x;
}

// CHECK-LABEL: define dso_local x86_fastcallcc void @"?qux@B@@QAI?AUA@@U2@@Z"
// CHECK:       (ptr inreg noundef %this, ptr inreg noalias sret(%struct.A) align 4 %agg.result, ptr inalloca(<{ %struct.A }>) %0)
// CHECK:   ret void

int main() {
  B b;
  A a = b.foo(A());
  a = b.bar(a);
  a = b.baz(a);
  a = b.qux(a);
}

// CHECK: call x86_thiscallcc ptr @"?foo@B@@QAE?AUA@@U2@@Z"
// CHECK:       (ptr noundef %{{[^,]*}}, ptr inalloca(<{ ptr, %struct.A }>) %{{[^,]*}})
// CHECK: call ptr @"?bar@B@@QAA?AUA@@U2@@Z"
// CHECK:       (ptr inalloca(<{ ptr, ptr, %struct.A }>) %{{[^,]*}})
// CHECK: call x86_stdcallcc ptr @"?baz@B@@QAG?AUA@@U2@@Z"
// CHECK:       (ptr inalloca(<{ ptr, ptr, %struct.A }>) %{{[^,]*}})
// CHECK: call x86_fastcallcc void @"?qux@B@@QAI?AUA@@U2@@Z"
// CHECK:       (ptr inreg noundef %{{[^,]*}}, ptr inreg sret(%struct.A) align 4 %{{.*}}, ptr inalloca(<{ %struct.A }>) %{{[^,]*}})
