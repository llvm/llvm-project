// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s -strict-whitespace

#define M1(x) x
#define M2 1;
void foo() {
  M1(
    M2);
  // CHECK: {{.*}}:7:{{[0-9]+}}: warning: expression result unused
  // CHECK: {{.*}}:4:{{[0-9]+}}: note: expanded from macro 'M2'
  // CHECK: {{.*}}:3:{{[0-9]+}}: note: expanded from macro 'M1'
}

#define A 1
#define B A
#define C B
void bar() {
  C;
  // CHECK: {{.*}}:17:3: warning: expression result unused
  // CHECK: {{.*}}:15:11: note: expanded from macro 'C'
  // CHECK: {{.*}}:14:11: note: expanded from macro 'B'
  // CHECK: {{.*}}:13:11: note: expanded from macro 'A'
}

// rdar://7597492
#define sprintf(str, A, B) \
__builtin___sprintf_chk (str, 0, 42, A, B)

void baz(char *Msg) {
  sprintf(Msg,  "  sizeof FoooLib            : =%3u\n",   12LL);
}


// PR9279: comprehensive tests for multi-level macro back traces
#define macro_args1(x) x
#define macro_args2(x) macro_args1(x)
#define macro_args3(x) macro_args2(x)

#define macro_many_args1(x, y, z) y
#define macro_many_args2(x, y, z) macro_many_args1(x, y, z)
#define macro_many_args3(x, y, z) macro_many_args2(x, y, z)

void test() {
  macro_args3(11);
  // CHECK: {{.*}}:43:15: warning: expression result unused
  // Also check that the 'caret' printing agrees with the location here where
  // its easy to FileCheck.
  // CHECK-NEXT:      macro_args3(11);
  // CHECK-NEXT: {{^              \^~}}
  // CHECK: {{.*}}:36:36: note: expanded from macro 'macro_args3'
  // CHECK: {{.*}}:35:36: note: expanded from macro 'macro_args2'
  // CHECK: {{.*}}:34:24: note: expanded from macro 'macro_args1'

  macro_many_args3(
    1,
    2,
    3);
  // CHECK: {{.*}}:55:5: warning: expression result unused
  // CHECK: {{.*}}:40:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:39:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:38:35: note: expanded from macro 'macro_many_args1'

  macro_many_args3(
    1,
    M2,
    3);
  // CHECK: {{.*}}:64:5: warning: expression result unused
  // CHECK: {{.*}}:4:12: note: expanded from macro 'M2'
  // CHECK: {{.*}}:40:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:39:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:38:35: note: expanded from macro 'macro_many_args1'

  macro_many_args3(
    1,
    macro_args2(22),
    3);
  // CHECK: {{.*}}:74:17: warning: expression result unused
  // This caret location needs to be printed *inside* a different macro's
  // arguments.
  // CHECK-NEXT:        macro_args2(22),
  // CHECK-NEXT: {{^                \^~}}
  // CHECK: {{.*}}:35:36: note: expanded from macro 'macro_args2'
  // CHECK: {{.*}}:34:24: note: expanded from macro 'macro_args1'
  // CHECK: {{.*}}:40:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:39:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:38:35: note: expanded from macro 'macro_many_args1'
}

#define variadic_args1(x, y, ...) y
#define variadic_args2(x, ...) variadic_args1(x, __VA_ARGS__)
#define variadic_args3(x, y, ...) variadic_args2(x, y, __VA_ARGS__)

void test2() {
  variadic_args3(1, 22, 3, 4);
  // CHECK: {{.*}}:93:21: warning: expression result unused
  // CHECK-NEXT:      variadic_args3(1, 22, 3, 4);
  // CHECK-NEXT: {{^                    \^~}}
  // CHECK: {{.*}}:90:53: note: expanded from macro 'variadic_args3'
  // CHECK: {{.*}}:89:50: note: expanded from macro 'variadic_args2'
  // CHECK: {{.*}}:88:35: note: expanded from macro 'variadic_args1'
}

#define variadic_pasting_args1(x, y, z) y
#define variadic_pasting_args2(x, ...) variadic_pasting_args1(x ## __VA_ARGS__)
#define variadic_pasting_args2a(x, y, ...) variadic_pasting_args1(x, y ## __VA_ARGS__)
#define variadic_pasting_args3(x, y, ...) variadic_pasting_args2(x, y, __VA_ARGS__)
#define variadic_pasting_args3a(x, y, ...) variadic_pasting_args2a(x, y, __VA_ARGS__)

void test3() {
  variadic_pasting_args3(1, 2, 3, 4);
  // CHECK: {{.*}}:109:32: warning: expression result unused
  // CHECK: {{.*}}:105:72: note: expanded from macro 'variadic_pasting_args3'
  // CHECK: {{.*}}:103:68: note: expanded from macro 'variadic_pasting_args2'
  // CHECK: {{.*}}:102:41: note: expanded from macro 'variadic_pasting_args1'

  variadic_pasting_args3a(1, 2, 3, 4);
  // CHECK: {{.*}}:115:30: warning: expression result unused
  // CHECK: {{.*}}:106:71: note: expanded from macro 'variadic_pasting_args3a'
  // CHECK: {{.*}}:104:70: note: expanded from macro 'variadic_pasting_args2a'
  // CHECK: {{.*}}:102:41: note: expanded from macro 'variadic_pasting_args1'
}

#define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
int test4 = BAD_CONDITIONAL_OPERATOR+BAD_CONDITIONAL_OPERATOR;
// CHECK:         {{.*}}:122:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                      \^}}
// CHECK:         {{.*}}:122:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                      \^}}
// CHECK:         {{.*}}:122:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                 ~~~~~\^~~~}}

#define QMARK ?
#define TWOL (2<
#define X 1+TWOL 3) QMARK 4:5
int x = X;
// CHECK:         {{.*}}:137:9: note: place parentheses around the '+' expression to silence this warning
// CHECK-NEXT:    int x = X;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:136:21: note: expanded from macro 'X'
// CHECK-NEXT:    #define X 1+TWOL 3) QMARK 4:5
// CHECK-NEXT: {{^          ~~~~~~~~~ \^}}
// CHECK-NEXT:    {{.*}}:134:15: note: expanded from macro 'QMARK'
// CHECK-NEXT:    #define QMARK ?
// CHECK-NEXT: {{^              \^}}
// CHECK-NEXT:    {{.*}}:137:9: note: place parentheses around the '?:' expression to evaluate it first
// CHECK-NEXT:    int x = X;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:136:21: note: expanded from macro 'X'
// CHECK-NEXT:    #define X 1+TWOL 3) QMARK 4:5
// CHECK-NEXT: {{^            ~~~~~~~~\^~~~~~~~~}}

#define ONEPLUS 1+
#define Y ONEPLUS (2<3) QMARK 4:5
int y = Y;
// CHECK:         {{.*}}:156:9: warning: operator '?:' has lower precedence than '+'; '+' will be evaluated first
// CHECK-NEXT:    int y = Y;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:155:25: note: expanded from macro 'Y'
// CHECK-NEXT:    #define Y ONEPLUS (2<3) QMARK 4:5
// CHECK-NEXT: {{^          ~~~~~~~~~~~~~ \^}}
// CHECK-NEXT:    {{.*}}:134:15: note: expanded from macro 'QMARK'
// CHECK-NEXT:    #define QMARK ?
// CHECK-NEXT: {{^              \^}}
