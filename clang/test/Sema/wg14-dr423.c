// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// expected-no-diagnostics

void GH39595(void) {
  // Ensure that qualifiers on function return types are dropped as part of the
  // declaration.
  extern const int const_int(void);
  // CHECK: FunctionDecl {{.*}} parent {{.*}} <col:3, col:34> col:20 referenced const_int 'int (void)' extern
  extern _Atomic int atomic(void);
  // CHECK: FunctionDecl {{.*}} parent {{.*}} <col:3, col:33> col:22 referenced atomic 'int (void)' extern

  (void)_Generic(const_int(), int : 1);
  (void)_Generic(atomic(), int : 1);

  // Make sure they're dropped from function pointers as well.
  _Atomic int (*fp)(void);
  (void)_Generic(fp(), int : 1);
}

void casting(void) {
  // Ensure that qualifiers on cast operations are also dropped.
  (void)_Generic((const int)12, int : 1);

  struct S { int i; } s;
  (void)_Generic((const struct S)s, struct S : 1);

  int i;
  __typeof__((const int)i) j;
  j = 100; // If we didn't strip the qualifiers during the cast, this would err.
}
