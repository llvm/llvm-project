// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// Passing a non-trivial C++ class type (non-trivial copy/move ctor or dtor) to
// os_log used to crash CodeGen: such arguments take the VarArgKind::Undefined
// path in checkFormatExpr, which only emitted the -Wnon-pod-varargs warning and
// let compilation proceed into CodeGen, where the argument's size has no
// corresponding integer type. Emit a hard error instead.

struct NonTrivial {
  char a, b, c;
  ~NonTrivial();
};

void test(void *buf, NonTrivial nt) {
  __builtin_os_log_format(buf, "%s", nt); // expected-error {{format specifies type 'char *' but the argument has type 'NonTrivial'}}
}
