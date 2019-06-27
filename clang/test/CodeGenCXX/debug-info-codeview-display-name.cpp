// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -std=c++98 | \
// RUN:    grep 'DISubprogram\|DICompositeType' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s --check-prefix=CHECK --check-prefix=UNQUAL
// RUN: %clang_cc1 -fblocks -debug-info-kind=line-tables-only -gcodeview -emit-llvm %s \
// RUN:       -o - -triple=x86_64-pc-win32 -std=c++98 | \
// RUN:    grep 'DISubprogram' | sed -e 's/.*name: "\([^"]*\)".*/"\1"/' | \
// RUN:    FileCheck %s --check-prefix=CHECK --check-prefix=QUAL

void freefunc() { }
// CHECK-DAG: "freefunc"

namespace N {
  int b() { return 0; }
// UNQUAL-DAG: "b"
// QUAL-DAG: "N::b"
  namespace { void func() { } }
// UNQUAL-DAG: "func"
// QUAL-DAG: "N::`anonymous namespace'::func"
}

void _c(void) {
  N::func();
}
// CHECK-DAG: "_c"

struct foo {
  int operator+(int);
  foo(){}
// UNQUAL-DAG: "foo"
// QUAL-DAG: "foo::foo"

  ~foo(){}
// UNQUAL-DAG: "~foo"
// QUAL-DAG: "foo::~foo"

  foo(int i){}
// UNQUAL-DAG: "foo"
// QUAL-DAG: "foo::foo"

  foo(char *q){}
// UNQUAL-DAG: "foo"
// QUAL-DAG: "foo::foo"

  static foo* static_method() { return 0; }
// UNQUAL-DAG: "static_method"
// QUAL-DAG: "foo::static_method"

};

void use_foo() {
  foo f1, f2(1), f3((char*)0);
  foo::static_method();
}

// UNQUAL-DAG: "operator+"
// QUAL-DAG: "foo::operator+"
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

// UNQUAL-DAG: "operator new"
// UNQUAL-DAG: "operator new[]"
// UNQUAL-DAG: "operator delete"
// UNQUAL-DAG: "operator delete[]"
// UNQUAL-DAG: "operator+"
// QUAL-DAG: "OverloadedNewDelete::operator new"
// QUAL-DAG: "OverloadedNewDelete::operator new[]"
// QUAL-DAG: "OverloadedNewDelete::operator delete"
// QUAL-DAG: "OverloadedNewDelete::operator delete[]"
// QUAL-DAG: "OverloadedNewDelete::operator+"


template <typename T, void (*)(void)>
void fn_tmpl() {}

template void fn_tmpl<int, freefunc>();
// CHECK-DAG: "fn_tmpl<int,&freefunc>"

template <typename A, typename B, typename C> struct ClassTemplate { A a; B b; C c; };
ClassTemplate<char, short, ClassTemplate<int, int, int> > f;
// This will only show up in normal debug builds.
// UNQUAL-DAG: "ClassTemplate<char,short,ClassTemplate<int,int,int> >"
