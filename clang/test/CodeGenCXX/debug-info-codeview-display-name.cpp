// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -Wno-new-returns-null -std=c++98 | \
// RUN:    grep -E 'DISubprogram|DICompositeType' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s --check-prefixes=CHECK,UNQUAL
// RUN: %clang_cc1 -fblocks -debug-info-kind=line-tables-only -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -Wno-new-returns-null -std=c++98 | \
// RUN:    grep 'DISubprogram' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s
// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -Wno-new-returns-null -std=c++11 | \
// RUN:    grep -E 'DISubprogram|DICompositeType' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s --check-prefixes=CHECK,UNQUAL
// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -Wno-new-returns-null | \
// RUN:    grep -E 'DISubprogram|DICompositeType' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s --check-prefixes=CHECK,UNQUAL

void freefunc() { }
// CHECK-DAG: "freefunc"

namespace N {
  int b() { return 0; }
// CHECK-DAG: "b"
  namespace { void func() { } }
// CHECK-DAG: "func"
}

void _c(void) {
  N::func();
}
// CHECK-DAG: "_c"

struct foo {
  int operator+(int);
  foo(){}
// CHECK-DAG: "foo"

  ~foo(){}
// CHECK-DAG: "~foo"

  foo(int i){}
// CHECK-DAG: "foo"

  foo(char *q){}
// CHECK-DAG: "foo"

  static foo* static_method() { return 0; }
// CHECK-DAG: "static_method"

};

void use_foo() {
  foo f1, f2(1), f3((char*)0);
  foo::static_method();
}

// CHECK-DAG: "operator+"
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

// CHECK-DAG: "operator new"
// CHECK-DAG: "operator new[]"
// CHECK-DAG: "operator delete"
// CHECK-DAG: "operator delete[]"
// CHECK-DAG: "operator+"

template <typename T, void (*)(void)>
void fn_tmpl() {}

template void fn_tmpl<int, freefunc>();
// CHECK-DAG: "fn_tmpl<int,&freefunc>"

template <typename T, void (*)(void)>
void fn_tmpl_typecheck() {}

template void fn_tmpl_typecheck<int, &freefunc>();
// CHECK-DAG: "fn_tmpl_typecheck<int,&freefunc>"

template <typename A, typename B, typename C> struct ClassTemplate { A a; B b; C c; };
ClassTemplate<char, short, ClassTemplate<int, int, int> > f;
// This will only show up in normal debug builds.  The space in `> >` is
// important for compatibility with Windows debuggers, so it should always be
// there when generating CodeView.
// UNQUAL-DAG: "ClassTemplate<char,short,ClassTemplate<int,int,int> >"
