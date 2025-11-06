// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name statement-expression.c %s

// No crash for the following examples, where GNU Statement Expression extension
// could introduce region terminators (break, goto etc) before implicit
// initializers in a struct or an array.
// See https://github.com/llvm/llvm-project/pull/89564

struct Foo {
  int field1;
  int field2;
};

void f1(void) {
  struct Foo foo = {
    .field1 = ({
      switch (0) {
      case 0:
        break; // A region terminator
      }
      0;
    }),
    // ImplicitValueInitExpr introduced here for .field2
  };
}

void f2(void) {
  int arr[3] = {
    [0] = ({
        goto L0; // A region terminator
L0:
      0;
    }),
    // ImplicitValueInitExpr introduced here for subscript [1]
    [2] = 0,
  };
}
