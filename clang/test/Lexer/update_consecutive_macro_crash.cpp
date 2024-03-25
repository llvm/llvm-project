// RUN: %clang -cc1 -fsyntax-only -verify %s 2>&1

#define X(val2) Y(val2++) // expected-note {{macro 'X' defined here}}
#define Y(expression) expression ;

void foo() {
  // https://github.com/llvm/llvm-project/issues/60722:
  //
  // - Due to to the error recovery, the lexer inserts a pair of () around the
  //   macro argument int{,}, so we will see [(, int, {, ,, }, )] tokens.
  // - however, the size of file id for the macro argument only takes account
  //   the written tokens  int{,} , and the extra inserted ) token points to the
  //    Limit source location which triggered an empty Partition violation.
  X(int{,}); // expected-error {{too many arguments provided to function-like macro invocation}} \
                 expected-error {{expected expression}} \
                 expected-note {{parentheses are required around macro argument containing braced initializer list}}
}
