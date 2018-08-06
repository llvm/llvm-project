// RUN: %clang_cc1 -fno-rtti-data -triple x86_64-windows-msvc -emit-llvm-only %s -verify

// Verify that we error out on this return adjusting thunk that we can't emit.

struct Incomplete;

struct A {
  virtual A * clone(Incomplete p) = 0;
};
struct B : virtual A {
  // expected-error@+1 2 {{cannot compile this return-adjusting thunk with incomplete parameter type yet}}
  B * clone(Incomplete p) override;
};
struct C : B { int c; };
C c;
