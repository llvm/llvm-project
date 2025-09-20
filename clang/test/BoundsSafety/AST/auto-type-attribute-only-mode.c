// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck --check-prefix=C %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck --check-prefixes=C,CXX %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck --check-prefix=C %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck --check-prefixes=C,CXX %s

#include <ptrcheck.h>

// Check that we drop the sugar.

// C:      FunctionDecl {{.+}} foo 'void (int * __counted_by(len), int)'
// C-NEXT: |-ParmVarDecl {{.+}} used ptr 'int * __counted_by(len)':'int *'
// C-NEXT: |-ParmVarDecl {{.+}} used len 'int'
// C-NEXT: | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
void foo(int *__counted_by(len) ptr, int len) {
  // C: VarDecl {{.+}} p 'int *' cinit
  __auto_type p = ptr;

#ifdef __cplusplus
  // CXX: VarDecl {{.+}} q 'int *' cinit
  auto q = ptr;

  // CXX: VarDecl {{.+}} r 'int *' cinit
  decltype(auto) r = ptr;
#endif
}

#ifdef __cplusplus
template<typename T>
struct cxx_dep {
  int len;
  T __counted_by(len) ptr;

  void f() {
    __auto_type p = ptr;
    auto q = ptr;
    decltype(auto) r = ptr;
  }
};

// CXX: ClassTemplateSpecializationDecl {{.+}} struct cxx_dep
// CXX: VarDecl {{.+}} p 'int *' cinit
// CXX: VarDecl {{.+}} q 'int *' cinit
// CXX: VarDecl {{.+}} r 'int *' cinit
template struct cxx_dep<int *>;
#endif

// C:      FunctionDecl {{.+}} bar 'void (int * __counted_by(42))'
// C-NEXT: |-ParmVarDecl {{.+}} used ptr 'int * __counted_by(42)':'int *'
void bar(int *__counted_by(42) ptr) {
  // C: VarDecl {{.+}} p 'int *' cinit
  __auto_type p = ptr;

#ifdef __cplusplus
  // CXX: VarDecl {{.+}} q 'int *' cinit
  auto q = ptr;

  // CXX: VarDecl {{.+}} r 'int *' cinit
  decltype(auto) r = ptr;
#endif
}

// C: FunctionDecl {{.+}} used get_magic_bytes 'void * __sized_by(5)({{.*}})'
void *__sized_by(5) get_magic_bytes(void);

void baz(void) {
  // C: VarDecl {{.+}} p 'void *' cinit
  __auto_type p = get_magic_bytes();

#ifdef __cplusplus
  // CXX: VarDecl {{.+}} q 'void *' cinit
  auto q = get_magic_bytes();

  // CXX: VarDecl {{.+}} r 'void *' cinit
  decltype(auto) r = get_magic_bytes();
#endif
}

#ifdef __cplusplus
template<typename T>
struct cxx_const {
  T __counted_by(7) ptr;

  void f() {
    __auto_type p = ptr;
    auto q = ptr;
    decltype(auto) r = ptr;
  }
};

// CXX: ClassTemplateSpecializationDecl {{.+}} struct cxx_const
// CXX: VarDecl {{.+}} p 'int *' cinit
// CXX: VarDecl {{.+}} q 'int *' cinit
// CXX: VarDecl {{.+}} r 'int *' cinit
template struct cxx_const<int *>;

template<typename T, int len>
struct cxx_const2 {
  T __counted_by(len) ptr;

  void f() {
    __auto_type p = ptr;
    auto q = ptr;
    decltype(auto) r = ptr;
  }
};

// CXX: ClassTemplateSpecializationDecl {{.+}} struct cxx_const2
// CXX: VarDecl {{.+}} p 'int *' cinit
// CXX: VarDecl {{.+}} q 'int *' cinit
// CXX: VarDecl {{.+}} r 'int *' cinit
template struct cxx_const2<int *, 123>;

// CXX: ClassTemplateSpecializationDecl {{.+}} struct cxx_const2
// CXX: VarDecl {{.+}} p 'unsigned int *' cinit
// CXX: VarDecl {{.+}} q 'unsigned int *' cinit
// CXX: VarDecl {{.+}} r 'unsigned int *' cinit
template struct cxx_const2<unsigned *, 99>;
#endif
