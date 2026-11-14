// A default argument instantiated for a parameter declared in an earlier PCH
// is written to the dependent PCH as a declaration update. The update must keep
// the initializer's ExprWithCleanups: users of the chain
// (Sema::CheckCXXDefaultArgExpr) rely on it to mark the full-expression of
// their own calls as needing cleanups. Without it the temporary's destructor
// cleanup outlives the full-expression, and for a guarded static local the
// guard-abort cleanup is left active for the rest of the function.
//
// The copy-initialization below relies on C++17 guaranteed copy elision:
// before that, the S<int> temporary has cleanups of its own that mask the
// missing marker. Direct-initialization shows the bug in every standard.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fexceptions \
// RUN:   -fcxx-exceptions -emit-pch -x c++-header %t/a.h -o %t/a.pch
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fexceptions \
// RUN:   -fcxx-exceptions -include-pch %t/a.pch -emit-pch -x c++-header \
// RUN:   %t/b.h -o %t/b.pch
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fexceptions \
// RUN:   -fcxx-exceptions -include-pch %t/b.pch -ast-dump \
// RUN:   -ast-dump-filter=use_default_arg %t/use.cpp | \
// RUN:   FileCheck --check-prefix=CHECK-AST %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fexceptions \
// RUN:   -fcxx-exceptions -include-pch %t/b.pch -emit-llvm -o - %t/use.cpp | \
// RUN:   FileCheck --check-prefix=CHECK-IR %s

//--- a.h
struct Alloc {
  Alloc();
  ~Alloc();
};
template <typename T>
struct S {
  S(const char *, const Alloc & = Alloc());
  ~S();
};
// Instantiate S<int> here, but not the default argument, so that the
// parameter declaration belongs to a.pch.
inline unsigned sizeof_S_int() { return sizeof(S<int>); }

//--- b.h
// Instantiates the default argument of a parameter owned by a.pch; the
// instantiation is written to b.pch as a declaration update.
inline void instantiate_default_arg() { S<int> s("in-b"); }

//--- use.cpp
void may_throw();
S<int> &use_default_arg() {
  static const S<int> s = "in-use";
  may_throw();
  return const_cast<S<int> &>(s);
}
void use_local() {
  S<int> s("local");
  may_throw();
}

// CHECK-AST: VarDecl {{.*}} s 'const S<int>' static cinit
// CHECK-AST-NEXT: ExprWithCleanups

// Static local: the temporary is destroyed inside the guarded region and the
// guard-abort cleanup ends with it, so the next statement needs no landing pad.
// CHECK-IR-LABEL: define {{.*}} @_Z15use_default_argv(
// CHECK-IR:       call void @_ZN5AllocD1Ev(
// CHECK-IR-NEXT:  call i32 @__cxa_atexit(
// CHECK-IR-NEXT:  call void @__cxa_guard_release(
// CHECK-IR:       call void @_Z9may_throwv()
// CHECK-IR-NEXT:  ret
//
// Plain local: the temporary is destroyed at the end of the declaration,
// before the next statement.
// CHECK-IR-LABEL: define {{.*}} @_Z9use_localv(
// CHECK-IR:       call void @_ZN5AllocD1Ev(
// CHECK-IR-NEXT:  invoke void @_Z9may_throwv()
