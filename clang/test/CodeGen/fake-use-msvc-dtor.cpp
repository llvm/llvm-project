// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -emit-llvm -fextend-variable-liveness=all -O1 -mconstructor-aliases -o /dev/null
// Verify that we do not crash when generating fake uses for parameters
// of destructor declarations that have no body (the should_call_delete
// implicit parameter in MSVC deleting destructors).

struct A {
  virtual ~A(void);
};
struct B {
  virtual ~B(void);
};

struct C : A, B {};

void foo() {
  C c;
}
