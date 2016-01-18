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

#define A(x) x
#define B(x) A(x)
#define C(x) B(x)
void bar() {
  C(1);
  // CHECK: {{.*}}:17:5: warning: expression result unused
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
  // CHECK: {{.*}}:40:15: warning: expression result unused
  // Also check that the 'caret' printing agrees with the location here where
  // its easy to FileCheck.
  // CHECK-NEXT:      macro_args3(11);
  // CHECK-NEXT: {{^              \^~}}

  macro_many_args3(
    1,
    2,
    3);
  // CHECK: {{.*}}:49:5: warning: expression result unused
  // CHECK: {{.*}}:37:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:36:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:35:35: note: expanded from macro 'macro_many_args1'

  macro_many_args3(
    1,
    M2,
    3);
  // CHECK: {{.*}}:58:5: warning: expression result unused
  // CHECK: {{.*}}:4:12: note: expanded from macro 'M2'
  // CHECK: {{.*}}:37:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:36:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:35:35: note: expanded from macro 'macro_many_args1'

  macro_many_args3(
    1,
    macro_args2(22),
    3);
  // CHECK: {{.*}}:68:17: warning: expression result unused
  // This caret location needs to be printed *inside* a different macro's
  // arguments.
  // CHECK-NEXT:        macro_args2(22),
  // CHECK-NEXT: {{^                \^~}}
  // CHECK: {{.*}}:32:36: note: expanded from macro 'macro_args2'
  // CHECK: {{.*}}:31:24: note: expanded from macro 'macro_args1'
  // CHECK: {{.*}}:37:55: note: expanded from macro 'macro_many_args3'
  // CHECK: {{.*}}:36:55: note: expanded from macro 'macro_many_args2'
  // CHECK: {{.*}}:35:35: note: expanded from macro 'macro_many_args1'
}

#define variadic_args1(x, y, ...) y
#define variadic_args2(x, ...) variadic_args1(x, __VA_ARGS__)
#define variadic_args3(x, y, ...) variadic_args2(x, y, __VA_ARGS__)

void test2() {
  variadic_args3(1, 22, 3, 4);
  // CHECK: {{.*}}:87:21: warning: expression result unused
  // CHECK-NEXT:      variadic_args3(1, 22, 3, 4);
  // CHECK-NEXT: {{^                    \^~}}
  // CHECK: {{.*}}:84:53: note: expanded from macro 'variadic_args3'
  // CHECK: {{.*}}:83:50: note: expanded from macro 'variadic_args2'
  // CHECK: {{.*}}:82:35: note: expanded from macro 'variadic_args1'
}

#define variadic_pasting_args1(x, y, z) y
#define variadic_pasting_args2(x, ...) variadic_pasting_args1(x ## __VA_ARGS__)
#define variadic_pasting_args2a(x, y, ...) variadic_pasting_args1(x, y ## __VA_ARGS__)
#define variadic_pasting_args3(x, y, ...) variadic_pasting_args2(x, y, __VA_ARGS__)
#define variadic_pasting_args3a(x, y, ...) variadic_pasting_args2a(x, y, __VA_ARGS__)

void test3() {
  variadic_pasting_args3(1, 2, 3, 4);
  // CHECK: {{.*}}:103:32: warning: expression result unused
  // CHECK: {{.*}}:99:72: note: expanded from macro 'variadic_pasting_args3'
  // CHECK: {{.*}}:97:68: note: expanded from macro 'variadic_pasting_args2'
  // CHECK: {{.*}}:96:41: note: expanded from macro 'variadic_pasting_args1'

  variadic_pasting_args3a(1, 2, 3, 4);
  // CHECK:        {{.*}}:109:3: warning: expression result unused
  // CHECK-NEXT:     variadic_pasting_args3a(1, 2, 3, 4);
  // CHECK-NEXT: {{  \^~~~~~~~~~~~~~~~~~~~~~~}}
  // CHECK:        {{.*}}:100:44: note: expanded from macro 'variadic_pasting_args3a'
  // CHECK-NEXT:   #define variadic_pasting_args3a(x, y, ...) variadic_pasting_args2a(x, y, __VA_ARGS__)
  // CHECK-NEXT: {{                                           \^~~~~~~~~~~~~~~~~~~~~~~}}
  // CHECK:        {{.*}}:98:70: note: expanded from macro 'variadic_pasting_args2a'
  // CHECK-NEXT:   #define variadic_pasting_args2a(x, y, ...) variadic_pasting_args1(x, y ## __VA_ARGS__)
  // CHECK-NEXT: {{                                                                     \^~~~~~~~~~~~~~~~}}
  // CHECK:        {{.*}}:96:41: note: expanded from macro 'variadic_pasting_args1'
  // CHECK-NEXT:   #define variadic_pasting_args1(x, y, z) y
  // CHECK-NEXT: {{                                        \^}}
}

#define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
int test4 = BAD_CONDITIONAL_OPERATOR+BAD_CONDITIONAL_OPERATOR;
// CHECK:         {{.*}}:124:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                      \^}}
// CHECK:         {{.*}}:124:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                      \^}}
// CHECK:         {{.*}}:124:39: note: expanded from macro 'BAD_CONDITIONAL_OPERATOR'
// CHECK-NEXT:    #define BAD_CONDITIONAL_OPERATOR (2<3)?2:3
// CHECK-NEXT: {{^                                 ~~~~~\^~~~}}

#define QMARK ?
#define TWOL (2<
#define X 1+TWOL 3) QMARK 4:5
int x = X;
// CHECK:         {{.*}}:139:9: note: place parentheses around the '+' expression to silence this warning
// CHECK-NEXT:    int x = X;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:138:21: note: expanded from macro 'X'
// CHECK-NEXT:    #define X 1+TWOL 3) QMARK 4:5
// CHECK-NEXT: {{^          ~~~~~~~~~ \^}}
// CHECK-NEXT:    {{.*}}:136:15: note: expanded from macro 'QMARK'
// CHECK-NEXT:    #define QMARK ?
// CHECK-NEXT: {{^              \^}}
// CHECK-NEXT:    {{.*}}:139:9: note: place parentheses around the '?:' expression to evaluate it first
// CHECK-NEXT:    int x = X;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:138:21: note: expanded from macro 'X'
// CHECK-NEXT:    #define X 1+TWOL 3) QMARK 4:5
// CHECK-NEXT: {{^            ~~~~~~~~\^~~~~~~~~}}

#define ONEPLUS 1+
#define Y ONEPLUS (2<3) QMARK 4:5
int y = Y;
// CHECK:         {{.*}}:158:9: warning: operator '?:' has lower precedence than '+'; '+' will be evaluated first
// CHECK-NEXT:    int y = Y;
// CHECK-NEXT: {{^        \^}}
// CHECK-NEXT:    {{.*}}:157:25: note: expanded from macro 'Y'
// CHECK-NEXT:    #define Y ONEPLUS (2<3) QMARK 4:5
// CHECK-NEXT: {{^          ~~~~~~~~~~~~~ \^}}
// CHECK-NEXT:    {{.*}}:136:15: note: expanded from macro 'QMARK'
// CHECK-NEXT:    #define QMARK ?
// CHECK-NEXT: {{^              \^}}

// PR14399
void iequals(int,int,int);
void foo_aa(char* s)
{
#define /* */ BARC(c, /* */b, a) (a + b ? c : c)
  iequals(__LINE__, BARC(123, (456 < 345), 789), 8);
}
// CHECK:         {{.*}}:174:21: warning: operator '?:' has lower precedence than '+'
// CHECK-NEXT:      iequals(__LINE__, BARC(123, (456 < 345), 789), 8);
// CHECK-NEXT: {{^                    \^~~~~~~~~~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT:    {{.*}}:173:41: note: expanded from macro 'BARC'
// CHECK-NEXT:    #define /* */ BARC(c, /* */b, a) (a + b ? c : c)
// CHECK-NEXT: {{^                                  ~~~~~ \^}}

#define APPEND2(NUM, SUFF) -1 != NUM ## SUFF
#define APPEND(NUM, SUFF) APPEND2(NUM, SUFF)
#define UTARG_MAX_U APPEND (MAX_UINT, UL)
#define MAX_UINT 18446744073709551615
#if UTARG_MAX_U
#endif

// CHECK:         {{.*}}:187:5: warning: left side of operator converted from negative value to unsigned: -1 to 18446744073709551615
// CHECK-NEXT:    #if UTARG_MAX_U
// CHECK-NEXT: {{^    \^~~~~~~~~~~}}
// CHECK-NEXT:    {{.*}}:185:21: note: expanded from macro 'UTARG_MAX_U'
// CHECK-NEXT:    #define UTARG_MAX_U APPEND (MAX_UINT, UL)
// CHECK-NEXT: {{^                    \^~~~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT:    {{.*}}:184:27: note: expanded from macro 'APPEND'
// CHECK-NEXT:    #define APPEND(NUM, SUFF) APPEND2(NUM, SUFF)
// CHECK-NEXT: {{^                          \^~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT:    {{.*}}:183:31: note: expanded from macro 'APPEND2'
// CHECK-NEXT:    #define APPEND2(NUM, SUFF) -1 != NUM ## SUFF
// CHECK-NEXT: {{^                           ~~ \^  ~~~~~~~~~~~}}

unsigned long strlen_test(const char *s);
#define __darwin_obsz(object) __builtin_object_size (object, 1)
#define sprintf2(str, ...) \
  __builtin___sprintf_chk (str, 0, __darwin_obsz(str), __VA_ARGS__)
#define Cstrlen(a)  strlen_test(a)
#define Csprintf    sprintf2
void f(char* pMsgBuf, char* pKeepBuf) {
Csprintf(pMsgBuf,"\nEnter minimum anagram length (2-%1d): ", strlen_test(pKeepBuf));
// FIXME: Change test to use 'Cstrlen' instead of 'strlen_test' when macro printing is fixed.
}
// CHECK:         {{.*}}:210:62: warning: format specifies type 'int' but the argument has type 'unsigned long'
// CHECK-NEXT:    Csprintf(pMsgBuf,"\nEnter minimum anagram length (2-%1d): ", strlen_test(pKeepBuf));
// CHECK-NEXT: {{^                                                    ~~~      \^~~~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT: {{^                                                    %1lu}}
// CHECK-NEXT:    {{.*}}:208:21: note: expanded from macro 'Csprintf'
// CHECK-NEXT:    #define Csprintf    sprintf2
// CHECK-NEXT: {{^                    \^}}
// CHECK-NEXT:    {{.*}}:206:56: note: expanded from macro 'sprintf2'
// CHECK-NEXT:      __builtin___sprintf_chk (str, 0, __darwin_obsz(str), __VA_ARGS__)
// CHECK-NEXT: {{^                                                       \^~~~~~~~~~~}}
