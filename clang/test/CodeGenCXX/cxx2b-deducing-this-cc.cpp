// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple i386-windows-msvc -o - | FileCheck %s

namespace CC {

struct T {
   static void f(T&);
   void __cdecl g(this T&);
   void __thiscall h(this T&);
   void i(this T&);
};

void a() {
    T t;
    T::f(t);
}
// CHECK: define dso_local void @"?a@CC@@YAXXZ"{{.*}}
// CHECK: call void @"?f@T@CC@@SAXAAU12@@Z"{{.*}}

void b() {
    T t;
    t.g();
}
// CHECK: define dso_local void @"?b@CC@@YAXXZ"{{.*}}
// CHECK: call void @"?g@T@CC@@SAX_VAAU12@@Z"{{.*}}

void c() {
    T t;
    t.h();
}
// CHECK: define dso_local void @"?c@CC@@YAXXZ"{{.*}}
// CHECK: call x86_thiscallcc void @"?h@T@CC@@SEX_VAAU12@@Z"{{.*}}

void d() {
    T t;
    t.i();
}
// CHECK: define dso_local void @"?d@CC@@YAXXZ"{{.*}}
// CHECK: call void @"?i@T@CC@@SAX_VAAU12@@Z"{{.*}}

}
