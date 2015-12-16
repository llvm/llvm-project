// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gcodeview -emit-llvm %s -o - -triple=x86_64-pc-win32 -std=c++98 | \
// RUN:  grep 'DISubprogram' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | FileCheck %s

void freefunc() { }
// CHECK-DAG: "freefunc"

namespace N {
  int b() { return 0; }
// CHECK-DAG: "N::b"
  namespace { void func() { } }
// CHECK-DAG: "N::`anonymous namespace'::func
}

void _c(void) {
  N::func();
}
// CHECK-DAG: "_c"

struct foo {
  int operator+(int);
  foo(){}
// CHECK-DAG: "foo::foo"

  ~foo(){}
// CHECK-DAG: "foo::~foo"

  foo(int i){}
// CHECK-DAG: "foo::foo"

  foo(char *q){}
// CHECK-DAG: "foo::foo"

  static foo* static_method() { return 0; }
// CHECK-DAG: "foo::static_method"

};

void use_foo() {
  foo f1, f2(1), f3((char*)0);
  foo::static_method();
}

// CHECK-DAG: "foo::operator+"
int foo::operator+(int a) { return a; }

// PR17371
struct OverloadedNewDelete {
  // __cdecl
  void *operator new(__SIZE_TYPE__);
  void *operator new[](__SIZE_TYPE__);
  void operator delete(void *);
  void operator delete[](void *);
  // __thiscall
  int operator+(int);
};

void *OverloadedNewDelete::operator new(__SIZE_TYPE__ s) { return 0; }
void *OverloadedNewDelete::operator new[](__SIZE_TYPE__ s) { return 0; }
void OverloadedNewDelete::operator delete(void *) { }
void OverloadedNewDelete::operator delete[](void *) { }
int OverloadedNewDelete::operator+(int x) { return x; };

// CHECK-DAG: "OverloadedNewDelete::operator new"
// CHECK-DAG: "OverloadedNewDelete::operator new[]"
// CHECK-DAG: "OverloadedNewDelete::operator delete"
// CHECK-DAG: "OverloadedNewDelete::operator delete[]"
// CHECK-DAG: "OverloadedNewDelete::operator+"

template <void (*)(void)>
void fn_tmpl() {}

template void fn_tmpl<freefunc>();
// CHECK-DAG: "fn_tmpl"
