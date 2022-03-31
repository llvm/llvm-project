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

// CHECK-LABEL: define dso_local x86_thiscallcc %struct.A* @"?foo@B@@QAE?AUA@@U2@@Z"
// CHECK:       (%struct.B* noundef %this, <{ %struct.A*, %struct.A }>* inalloca(<{ %struct.A*, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ %struct.A*, %struct.A }>, <{ %struct.A*, %struct.A }>* %{{.*}}, i32 0, i32 0
// CHECK:   load %struct.A*, %struct.A**
// CHECK:   ret %struct.A*

A B::bar(A x) {
  return x;
}

// CHECK-LABEL: define dso_local %struct.A* @"?bar@B@@QAA?AUA@@U2@@Z"
// CHECK:       (<{ %struct.B*, %struct.A*, %struct.A }>* inalloca(<{ %struct.B*, %struct.A*, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ %struct.B*, %struct.A*, %struct.A }>, <{ %struct.B*, %struct.A*, %struct.A }>* %{{.*}}, i32 0, i32 1
// CHECK:   load %struct.A*, %struct.A**
// CHECK:   ret %struct.A*

A B::baz(A x) {
  return x;
}

// CHECK-LABEL: define dso_local x86_stdcallcc %struct.A* @"?baz@B@@QAG?AUA@@U2@@Z"
// CHECK:       (<{ %struct.B*, %struct.A*, %struct.A }>* inalloca(<{ %struct.B*, %struct.A*, %struct.A }>) %0)
// CHECK:   getelementptr inbounds <{ %struct.B*, %struct.A*, %struct.A }>, <{ %struct.B*, %struct.A*, %struct.A }>* %{{.*}}, i32 0, i32 1
// CHECK:   load %struct.A*, %struct.A**
// CHECK:   ret %struct.A*

A B::qux(A x) {
  return x;
}

// CHECK-LABEL: define dso_local x86_fastcallcc void @"?qux@B@@QAI?AUA@@U2@@Z"
// CHECK:       (%struct.B* inreg noundef %this, %struct.A* inreg noalias sret(%struct.A) align 4 %agg.result, <{ %struct.A }>* inalloca(<{ %struct.A }>) %0)
// CHECK:   ret void

int main() {
  B b;
  A a = b.foo(A());
  a = b.bar(a);
  a = b.baz(a);
  a = b.qux(a);
}

// CHECK: call x86_thiscallcc %struct.A* @"?foo@B@@QAE?AUA@@U2@@Z"
// CHECK:       (%struct.B* noundef %{{[^,]*}}, <{ %struct.A*, %struct.A }>* inalloca(<{ %struct.A*, %struct.A }>) %{{[^,]*}})
// CHECK: call %struct.A* @"?bar@B@@QAA?AUA@@U2@@Z"
// CHECK:       (<{ %struct.B*, %struct.A*, %struct.A }>* inalloca(<{ %struct.B*, %struct.A*, %struct.A }>) %{{[^,]*}})
// CHECK: call x86_stdcallcc %struct.A* @"?baz@B@@QAG?AUA@@U2@@Z"
// CHECK:       (<{ %struct.B*, %struct.A*, %struct.A }>* inalloca(<{ %struct.B*, %struct.A*, %struct.A }>) %{{[^,]*}})
// CHECK: call x86_fastcallcc void @"?qux@B@@QAI?AUA@@U2@@Z"
// CHECK:       (%struct.B* inreg noundef %{{[^,]*}}, %struct.A* inreg sret(%struct.A) align 4 %{{.*}}, <{ %struct.A }>* inalloca(<{ %struct.A }>) %{{[^,]*}})
