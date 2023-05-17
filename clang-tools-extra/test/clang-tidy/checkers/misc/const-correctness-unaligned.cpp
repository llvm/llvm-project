// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: [\
// RUN:   {key: 'misc-const-correctness.TransformValues', value: true}, \
// RUN:   {key: 'misc-const-correctness.WarnPointersAsValues', value: false}, \
// RUN:   {key: 'misc-const-correctness.TransformPointersAsValues', value: false}, \
// RUN:   ]}" -- -fno-delayed-template-parsing -fms-extensions

struct S {};

void f(__unaligned S *);

void scope() {
  // FIXME: This is a bug in the analysis, that is confused by '__unaligned'.
  // https://bugs.llvm.org/show_bug.cgi?id=51756
  S s;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 's' of type 'S' can be declared 'const'
  // CHECK-FIXES: S const s;
  f(&s);
}
