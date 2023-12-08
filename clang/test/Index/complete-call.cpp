// Note: the run lines follow their respective tests, since line/column
// matter in this test.

void foo_1();
void foo_2(int);
void foo_2(void *);
void foo_3(int, int);
void foo_3(void *, void *);
void foo_4(int, int);
void foo_4(void *, int);
void foo_5(int, int);
void foo_5(int, void *);
template<class T> void foo_6();
template<class T> void foo_7(T);
template<class T> void foo_8(T, T);
template<class T> void foo_9(int, T);
template<class T> void foo_9(void *, T);
template<class T> void foo_10(T, int, int);
template<class U> void foo_10(U, void *, void *);
template<class T, class U> void foo_11(T, U);
template<class T = int> void foo_12(T, T);
template<class V>
struct S {
  void foo_1();
  void foo_2(int);
  void foo_2(void *);
  void foo_3(int, int);
  void foo_3(void *, void *);
  void foo_4(int, int);
  void foo_4(void *, int);
  void foo_5(int, int);
  void foo_5(int, void *);
  template<class T> void foo_6();
  template<class T> void foo_7(T);
  template<class T> void foo_8(T, T);
  template<class T> void foo_9(int, T);
  template<class T> void foo_9(void *, T);
  template<class T> void foo_10(T, int, int);
  template<class U> void foo_10(U, void *, void *);
  template<class T, class U> void foo_11(T, U);
  template<class T = int> void foo_12(T, T);
  template<class T> void foo_13(V, T, T);
};

int main() {
  void *p = 0;
  foo_1();
  foo_2(42);
  foo_3(42, 42);
  foo_3(p, p);
  foo_4(42, 42);
  foo_4(p, 42);
  foo_5(42, 42);
  foo_6<int>();
  foo_7(42);
  foo_7<int>(42);
  foo_8(42, 42);
  foo_9(42, 42);
  foo_9(p, 42);
  foo_10(42, 42, 42);
  foo_11(42, 42);
  foo_11<int>(42, 42);
  foo_11<int, void *>(42, p);
  foo_12(p, p);

  S<int> s;
  s.foo_1();
  s.foo_2(42);
  s.foo_3(42, 42);
  s.foo_3(p, p);
  s.foo_4(42, 42);
  s.foo_4(p, 42);
  s.foo_5(42, 42);
  s.foo_6<int>();
  s.foo_7(42);
  s.foo_7<int>(42);
  s.foo_8(42, 42);
  s.foo_9(42, 42);
  s.foo_9(p, 42);
  s.foo_10(42, 42, 42);
  s.foo_11(42, 42);
  s.foo_11<int>(42, 42);
  s.foo_11<int, void *>(42, p);
  s.foo_12(p, p);
  s.foo_13(42, 42, 42);

  foo_1(42,);
  foo_2(42,);
  foo_6<int>(42,);
  foo_7(42,);
  s.foo_1(42,);
  s.foo_2(42,);
  s.foo_6<int>(42,);
  s.foo_7(42,);
}

struct Bar {
  static void foo_1();
  void foo_1(float);
  static void foo_1(int);
};

void test() {
  Bar::foo_1();
  Bar b;
  b.foo_1();
}

struct Bar2 : public Bar {
  Bar2() {
    Bar::foo_1();
  }
};

struct BarTemplates {
  static void foo_1() {}
  void foo_1(float) {}
  static void foo_1(int) {}

  template<class T1, class T2>
  static void foo_1(T1 a, T2 b) { a + b; }

  template<class T1, class T2>
  void foo_1(T1 a, T2 b, float c) { a + b + c; }

  template<class T1, class T2>
  static void foo_1(T2 a, int b, T1 c)  { a + b + c; }
};

void testTemplates() {
  BarTemplates::foo_1();
  BarTemplates b;
  b.foo_1();
}

struct Bar2Template : public BarTemplates {
  Bar2Template() {
    BarTemplates::foo_1();
  }
};

// RUN: c-index-test -code-completion-at=%s:47:9 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Any type
// CHECK-CC1-NEXT: Any value
// CHECK-CC1-NEXT: Enum tag
// CHECK-CC1-NEXT: Union tag
// CHECK-CC1-NEXT: Struct tag
// CHECK-CC1-NEXT: Class name
// CHECK-CC1-NEXT: Nested name specifier
// CHECK-CC1-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:48:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text foo_2}{LeftParen (}{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text foo_2}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Any type
// CHECK-CC2-NEXT: Any value
// CHECK-CC2-NEXT: Enum tag
// CHECK-CC2-NEXT: Union tag
// CHECK-CC2-NEXT: Struct tag
// CHECK-CC2-NEXT: Class name
// CHECK-CC2-NEXT: Nested name specifier
// CHECK-CC2-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:49:9 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC3: Completion contexts:
// CHECK-CC3-NEXT: Any type
// CHECK-CC3-NEXT: Any value
// CHECK-CC3-NEXT: Enum tag
// CHECK-CC3-NEXT: Union tag
// CHECK-CC3-NEXT: Struct tag
// CHECK-CC3-NEXT: Class name
// CHECK-CC3-NEXT: Nested name specifier
// CHECK-CC3-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:49:12 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC4: Completion contexts:
// CHECK-CC4-NEXT: Any type
// CHECK-CC4-NEXT: Any value
// CHECK-CC4-NEXT: Enum tag
// CHECK-CC4-NEXT: Union tag
// CHECK-CC4-NEXT: Struct tag
// CHECK-CC4-NEXT: Class name
// CHECK-CC4-NEXT: Nested name specifier
// CHECK-CC4-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:50:11 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC5: Completion contexts:
// CHECK-CC5-NEXT: Any type
// CHECK-CC5-NEXT: Any value
// CHECK-CC5-NEXT: Enum tag
// CHECK-CC5-NEXT: Union tag
// CHECK-CC5-NEXT: Struct tag
// CHECK-CC5-NEXT: Class name
// CHECK-CC5-NEXT: Nested name specifier
// CHECK-CC5-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:51:12 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: OverloadCandidate:{ResultType void}{Text foo_4}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC6: Completion contexts:
// CHECK-CC6-NEXT: Any type
// CHECK-CC6-NEXT: Any value
// CHECK-CC6-NEXT: Enum tag
// CHECK-CC6-NEXT: Union tag
// CHECK-CC6-NEXT: Struct tag
// CHECK-CC6-NEXT: Class name
// CHECK-CC6-NEXT: Nested name specifier
// CHECK-CC6-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:52:11 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: OverloadCandidate:{ResultType void}{Text foo_4}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC7: Completion contexts:
// CHECK-CC7-NEXT: Any type
// CHECK-CC7-NEXT: Any value
// CHECK-CC7-NEXT: Enum tag
// CHECK-CC7-NEXT: Union tag
// CHECK-CC7-NEXT: Struct tag
// CHECK-CC7-NEXT: Class name
// CHECK-CC7-NEXT: Nested name specifier
// CHECK-CC7-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:53:12 %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: OverloadCandidate:{ResultType void}{Text foo_5}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC8: OverloadCandidate:{ResultType void}{Text foo_5}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC8: Completion contexts:
// CHECK-CC8-NEXT: Any type
// CHECK-CC8-NEXT: Any value
// CHECK-CC8-NEXT: Enum tag
// CHECK-CC8-NEXT: Union tag
// CHECK-CC8-NEXT: Struct tag
// CHECK-CC8-NEXT: Class name
// CHECK-CC8-NEXT: Nested name specifier
// CHECK-CC8-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:54:14 %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: OverloadCandidate:{ResultType void}{Text foo_6}{LeftParen (}{RightParen )} (1)
// CHECK-CC9: Completion contexts:
// CHECK-CC9-NEXT: Any type
// CHECK-CC9-NEXT: Any value
// CHECK-CC9-NEXT: Enum tag
// CHECK-CC9-NEXT: Union tag
// CHECK-CC9-NEXT: Struct tag
// CHECK-CC9-NEXT: Class name
// CHECK-CC9-NEXT: Nested name specifier
// CHECK-CC9-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:55:9 %s | FileCheck -check-prefix=CHECK-CC10 %s
// CHECK-CC10: OverloadCandidate:{ResultType void}{Text foo_7}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC10: Completion contexts:
// CHECK-CC10-NEXT: Any type
// CHECK-CC10-NEXT: Any value
// CHECK-CC10-NEXT: Enum tag
// CHECK-CC10-NEXT: Union tag
// CHECK-CC10-NEXT: Struct tag
// CHECK-CC10-NEXT: Class name
// CHECK-CC10-NEXT: Nested name specifier
// CHECK-CC10-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:56:14 %s | FileCheck -check-prefix=CHECK-CC11 %s
// CHECK-CC11: OverloadCandidate:{ResultType void}{Text foo_7}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC11: Completion contexts:
// CHECK-CC11-NEXT: Any type
// CHECK-CC11-NEXT: Any value
// CHECK-CC11-NEXT: Enum tag
// CHECK-CC11-NEXT: Union tag
// CHECK-CC11-NEXT: Struct tag
// CHECK-CC11-NEXT: Class name
// CHECK-CC11-NEXT: Nested name specifier
// CHECK-CC11-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:57:12 %s | FileCheck -check-prefix=CHECK-CC12 %s
// CHECK-CC12: OverloadCandidate:{ResultType void}{Text foo_8}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC12: Completion contexts:
// CHECK-CC12-NEXT: Any type
// CHECK-CC12-NEXT: Any value
// CHECK-CC12-NEXT: Enum tag
// CHECK-CC12-NEXT: Union tag
// CHECK-CC12-NEXT: Struct tag
// CHECK-CC12-NEXT: Class name
// CHECK-CC12-NEXT: Nested name specifier
// CHECK-CC12-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:58:12 %s | FileCheck -check-prefix=CHECK-CC13 %s
// CHECK-CC13: OverloadCandidate:{ResultType void}{Text foo_9}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter T}{RightParen )} (1)
// CHECK-CC13: Completion contexts:
// CHECK-CC13-NEXT: Any type
// CHECK-CC13-NEXT: Any value
// CHECK-CC13-NEXT: Enum tag
// CHECK-CC13-NEXT: Union tag
// CHECK-CC13-NEXT: Struct tag
// CHECK-CC13-NEXT: Class name
// CHECK-CC13-NEXT: Nested name specifier
// CHECK-CC13-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:59:11 %s | FileCheck -check-prefix=CHECK-CC14 %s
// CHECK-CC14: OverloadCandidate:{ResultType void}{Text foo_9}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter T}{RightParen )} (1)
// CHECK-CC14: Completion contexts:
// CHECK-CC14-NEXT: Any type
// CHECK-CC14-NEXT: Any value
// CHECK-CC14-NEXT: Enum tag
// CHECK-CC14-NEXT: Union tag
// CHECK-CC14-NEXT: Struct tag
// CHECK-CC14-NEXT: Class name
// CHECK-CC14-NEXT: Nested name specifier
// CHECK-CC14-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:60:10 %s | FileCheck -check-prefix=CHECK-CC15 %s
// CHECK-CC15: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{CurrentParameter U}{Comma , }{Placeholder void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC15: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{CurrentParameter T}{Comma , }{Placeholder int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC15: Completion contexts:
// CHECK-CC15-NEXT: Any type
// CHECK-CC15-NEXT: Any value
// CHECK-CC15-NEXT: Enum tag
// CHECK-CC15-NEXT: Union tag
// CHECK-CC15-NEXT: Struct tag
// CHECK-CC15-NEXT: Class name
// CHECK-CC15-NEXT: Nested name specifier
// CHECK-CC15-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:60:13 %s | FileCheck -check-prefix=CHECK-CC16 %s
// CHECK-CC16: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC16: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC16: Completion contexts:
// CHECK-CC16-NEXT: Any type
// CHECK-CC16-NEXT: Any value
// CHECK-CC16-NEXT: Enum tag
// CHECK-CC16-NEXT: Union tag
// CHECK-CC16-NEXT: Struct tag
// CHECK-CC16-NEXT: Class name
// CHECK-CC16-NEXT: Nested name specifier
// CHECK-CC16-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:60:17 %s | FileCheck -check-prefix=CHECK-CC17 %s
// CHECK-CC17: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC17: Completion contexts:
// CHECK-CC17-NEXT: Any type
// CHECK-CC17-NEXT: Any value
// CHECK-CC17-NEXT: Enum tag
// CHECK-CC17-NEXT: Union tag
// CHECK-CC17-NEXT: Struct tag
// CHECK-CC17-NEXT: Class name
// CHECK-CC17-NEXT: Nested name specifier
// CHECK-CC17-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:61:10 %s | FileCheck -check-prefix=CHECK-CC18 %s
// CHECK-CC18: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{CurrentParameter T}{Comma , }{Placeholder U}{RightParen )} (1)
// CHECK-CC18: Completion contexts:
// CHECK-CC18-NEXT: Any type
// CHECK-CC18-NEXT: Any value
// CHECK-CC18-NEXT: Enum tag
// CHECK-CC18-NEXT: Union tag
// CHECK-CC18-NEXT: Struct tag
// CHECK-CC18-NEXT: Class name
// CHECK-CC18-NEXT: Nested name specifier
// CHECK-CC18-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:61:13 %s | FileCheck -check-prefix=CHECK-CC19 %s
// CHECK-CC19: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter U}{RightParen )} (1)
// CHECK-CC19: Completion contexts:
// CHECK-CC19-NEXT: Any type
// CHECK-CC19-NEXT: Any value
// CHECK-CC19-NEXT: Enum tag
// CHECK-CC19-NEXT: Union tag
// CHECK-CC19-NEXT: Struct tag
// CHECK-CC19-NEXT: Class name
// CHECK-CC19-NEXT: Nested name specifier
// CHECK-CC19-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:62:15 %s | FileCheck -check-prefix=CHECK-CC20 %s
// CHECK-CC20: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder U}{RightParen )} (1)
// CHECK-CC20: Completion contexts:
// CHECK-CC20-NEXT: Any type
// CHECK-CC20-NEXT: Any value
// CHECK-CC20-NEXT: Enum tag
// CHECK-CC20-NEXT: Union tag
// CHECK-CC20-NEXT: Struct tag
// CHECK-CC20-NEXT: Class name
// CHECK-CC20-NEXT: Nested name specifier
// CHECK-CC20-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:62:18 %s | FileCheck -check-prefix=CHECK-CC21 %s
// CHECK-CC21: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter U}{RightParen )} (1)
// CHECK-CC21: Completion contexts:
// CHECK-CC21-NEXT: Any type
// CHECK-CC21-NEXT: Any value
// CHECK-CC21-NEXT: Enum tag
// CHECK-CC21-NEXT: Union tag
// CHECK-CC21-NEXT: Struct tag
// CHECK-CC21-NEXT: Class name
// CHECK-CC21-NEXT: Nested name specifier
// CHECK-CC21-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:63:26 %s | FileCheck -check-prefix=CHECK-CC22 %s
// CHECK-CC22: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC22: Completion contexts:
// CHECK-CC22-NEXT: Any type
// CHECK-CC22-NEXT: Any value
// CHECK-CC22-NEXT: Enum tag
// CHECK-CC22-NEXT: Union tag
// CHECK-CC22-NEXT: Struct tag
// CHECK-CC22-NEXT: Class name
// CHECK-CC22-NEXT: Nested name specifier
// CHECK-CC22-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:64:10 %s | FileCheck -check-prefix=CHECK-CC23 %s
// CHECK-CC23: OverloadCandidate:{ResultType void}{Text foo_12}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC23: Completion contexts:
// CHECK-CC23-NEXT: Any type
// CHECK-CC23-NEXT: Any value
// CHECK-CC23-NEXT: Enum tag
// CHECK-CC23-NEXT: Union tag
// CHECK-CC23-NEXT: Struct tag
// CHECK-CC23-NEXT: Class name
// CHECK-CC23-NEXT: Nested name specifier
// CHECK-CC23-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:64:12 %s | FileCheck -check-prefix=CHECK-CC24 %s
// CHECK-CC24: OverloadCandidate:{ResultType void}{Text foo_12}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC24: Completion contexts:
// CHECK-CC24-NEXT: Any type
// CHECK-CC24-NEXT: Any value
// CHECK-CC24-NEXT: Enum tag
// CHECK-CC24-NEXT: Union tag
// CHECK-CC24-NEXT: Struct tag
// CHECK-CC24-NEXT: Class name
// CHECK-CC24-NEXT: Nested name specifier
// CHECK-CC24-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:67:11 %s | FileCheck -check-prefix=CHECK-CC25 %s
// CHECK-CC25: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC25: Completion contexts:
// CHECK-CC25-NEXT: Any type
// CHECK-CC25-NEXT: Any value
// CHECK-CC25-NEXT: Enum tag
// CHECK-CC25-NEXT: Union tag
// CHECK-CC25-NEXT: Struct tag
// CHECK-CC25-NEXT: Class name
// CHECK-CC25-NEXT: Nested name specifier
// CHECK-CC25-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:68:11 %s | FileCheck -check-prefix=CHECK-CC26 %s
// CHECK-CC26: OverloadCandidate:{ResultType void}{Text foo_2}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC26: OverloadCandidate:{ResultType void}{Text foo_2}{LeftParen (}{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC26: Completion contexts:
// CHECK-CC26-NEXT: Any type
// CHECK-CC26-NEXT: Any value
// CHECK-CC26-NEXT: Enum tag
// CHECK-CC26-NEXT: Union tag
// CHECK-CC26-NEXT: Struct tag
// CHECK-CC26-NEXT: Class name
// CHECK-CC26-NEXT: Nested name specifier
// CHECK-CC26-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:69:11 %s | FileCheck -check-prefix=CHECK-CC27 %s
// CHECK-CC27: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC27: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC27: Completion contexts:
// CHECK-CC27-NEXT: Any type
// CHECK-CC27-NEXT: Any value
// CHECK-CC27-NEXT: Enum tag
// CHECK-CC27-NEXT: Union tag
// CHECK-CC27-NEXT: Struct tag
// CHECK-CC27-NEXT: Class name
// CHECK-CC27-NEXT: Nested name specifier
// CHECK-CC27-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:69:14 %s | FileCheck -check-prefix=CHECK-CC28 %s
// CHECK-CC28: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC28: Completion contexts:
// CHECK-CC28-NEXT: Any type
// CHECK-CC28-NEXT: Any value
// CHECK-CC28-NEXT: Enum tag
// CHECK-CC28-NEXT: Union tag
// CHECK-CC28-NEXT: Struct tag
// CHECK-CC28-NEXT: Class name
// CHECK-CC28-NEXT: Nested name specifier
// CHECK-CC28-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:70:13 %s | FileCheck -check-prefix=CHECK-CC29 %s
// CHECK-CC29: OverloadCandidate:{ResultType void}{Text foo_3}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC29: Completion contexts:
// CHECK-CC29-NEXT: Any type
// CHECK-CC29-NEXT: Any value
// CHECK-CC29-NEXT: Enum tag
// CHECK-CC29-NEXT: Union tag
// CHECK-CC29-NEXT: Struct tag
// CHECK-CC29-NEXT: Class name
// CHECK-CC29-NEXT: Nested name specifier
// CHECK-CC29-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:71:14 %s | FileCheck -check-prefix=CHECK-CC30 %s
// CHECK-CC30: OverloadCandidate:{ResultType void}{Text foo_4}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC30: Completion contexts:
// CHECK-CC30-NEXT: Any type
// CHECK-CC30-NEXT: Any value
// CHECK-CC30-NEXT: Enum tag
// CHECK-CC30-NEXT: Union tag
// CHECK-CC30-NEXT: Struct tag
// CHECK-CC30-NEXT: Class name
// CHECK-CC30-NEXT: Nested name specifier
// CHECK-CC30-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:72:13 %s | FileCheck -check-prefix=CHECK-CC31 %s
// CHECK-CC31: OverloadCandidate:{ResultType void}{Text foo_4}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC31: Completion contexts:
// CHECK-CC31-NEXT: Any type
// CHECK-CC31-NEXT: Any value
// CHECK-CC31-NEXT: Enum tag
// CHECK-CC31-NEXT: Union tag
// CHECK-CC31-NEXT: Struct tag
// CHECK-CC31-NEXT: Class name
// CHECK-CC31-NEXT: Nested name specifier
// CHECK-CC31-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:73:14 %s | FileCheck -check-prefix=CHECK-CC32 %s
// CHECK-CC32: OverloadCandidate:{ResultType void}{Text foo_5}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC32: OverloadCandidate:{ResultType void}{Text foo_5}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC32: Completion contexts:
// CHECK-CC32-NEXT: Any type
// CHECK-CC32-NEXT: Any value
// CHECK-CC32-NEXT: Enum tag
// CHECK-CC32-NEXT: Union tag
// CHECK-CC32-NEXT: Struct tag
// CHECK-CC32-NEXT: Class name
// CHECK-CC32-NEXT: Nested name specifier
// CHECK-CC32-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:74:16 %s | FileCheck -check-prefix=CHECK-CC33 %s
// CHECK-CC33: OverloadCandidate:{ResultType void}{Text foo_6}{LeftParen (}{RightParen )} (1)
// CHECK-CC33: Completion contexts:
// CHECK-CC33-NEXT: Any type
// CHECK-CC33-NEXT: Any value
// CHECK-CC33-NEXT: Enum tag
// CHECK-CC33-NEXT: Union tag
// CHECK-CC33-NEXT: Struct tag
// CHECK-CC33-NEXT: Class name
// CHECK-CC33-NEXT: Nested name specifier
// CHECK-CC33-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:75:11 %s | FileCheck -check-prefix=CHECK-CC34 %s
// CHECK-CC34: OverloadCandidate:{ResultType void}{Text foo_7}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC34: Completion contexts:
// CHECK-CC34-NEXT: Any type
// CHECK-CC34-NEXT: Any value
// CHECK-CC34-NEXT: Enum tag
// CHECK-CC34-NEXT: Union tag
// CHECK-CC34-NEXT: Struct tag
// CHECK-CC34-NEXT: Class name
// CHECK-CC34-NEXT: Nested name specifier
// CHECK-CC34-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:76:16 %s | FileCheck -check-prefix=CHECK-CC35 %s
// CHECK-CC35: OverloadCandidate:{ResultType void}{Text foo_7}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC35: Completion contexts:
// CHECK-CC35-NEXT: Any type
// CHECK-CC35-NEXT: Any value
// CHECK-CC35-NEXT: Enum tag
// CHECK-CC35-NEXT: Union tag
// CHECK-CC35-NEXT: Struct tag
// CHECK-CC35-NEXT: Class name
// CHECK-CC35-NEXT: Nested name specifier
// CHECK-CC35-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:77:14 %s | FileCheck -check-prefix=CHECK-CC36 %s
// CHECK-CC36: OverloadCandidate:{ResultType void}{Text foo_8}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC36: Completion contexts:
// CHECK-CC36-NEXT: Any type
// CHECK-CC36-NEXT: Any value
// CHECK-CC36-NEXT: Enum tag
// CHECK-CC36-NEXT: Union tag
// CHECK-CC36-NEXT: Struct tag
// CHECK-CC36-NEXT: Class name
// CHECK-CC36-NEXT: Nested name specifier
// CHECK-CC36-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:78:14 %s | FileCheck -check-prefix=CHECK-CC37 %s
// CHECK-CC37: OverloadCandidate:{ResultType void}{Text foo_9}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter T}{RightParen )} (1)
// CHECK-CC37: Completion contexts:
// CHECK-CC37-NEXT: Any type
// CHECK-CC37-NEXT: Any value
// CHECK-CC37-NEXT: Enum tag
// CHECK-CC37-NEXT: Union tag
// CHECK-CC37-NEXT: Struct tag
// CHECK-CC37-NEXT: Class name
// CHECK-CC37-NEXT: Nested name specifier
// CHECK-CC37-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:79:13 %s | FileCheck -check-prefix=CHECK-CC38 %s
// CHECK-CC38: OverloadCandidate:{ResultType void}{Text foo_9}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter T}{RightParen )} (1)
// CHECK-CC38: Completion contexts:
// CHECK-CC38-NEXT: Any type
// CHECK-CC38-NEXT: Any value
// CHECK-CC38-NEXT: Enum tag
// CHECK-CC38-NEXT: Union tag
// CHECK-CC38-NEXT: Struct tag
// CHECK-CC38-NEXT: Class name
// CHECK-CC38-NEXT: Nested name specifier
// CHECK-CC38-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:80:12 %s | FileCheck -check-prefix=CHECK-CC39 %s
// CHECK-CC39: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{CurrentParameter T}{Comma , }{Placeholder int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC39: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{CurrentParameter U}{Comma , }{Placeholder void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC39: Completion contexts:
// CHECK-CC39-NEXT: Any type
// CHECK-CC39-NEXT: Any value
// CHECK-CC39-NEXT: Enum tag
// CHECK-CC39-NEXT: Union tag
// CHECK-CC39-NEXT: Struct tag
// CHECK-CC39-NEXT: Class name
// CHECK-CC39-NEXT: Nested name specifier
// CHECK-CC39-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:80:15 %s | FileCheck -check-prefix=CHECK-CC40 %s
// CHECK-CC40: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC40: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{Comma , }{Placeholder void *}{RightParen )} (1)
// CHECK-CC40: Completion contexts:
// CHECK-CC40-NEXT: Any type
// CHECK-CC40-NEXT: Any value
// CHECK-CC40-NEXT: Enum tag
// CHECK-CC40-NEXT: Union tag
// CHECK-CC40-NEXT: Struct tag
// CHECK-CC40-NEXT: Class name
// CHECK-CC40-NEXT: Nested name specifier
// CHECK-CC40-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:80:19 %s | FileCheck -check-prefix=CHECK-CC41 %s
// CHECK-CC41: OverloadCandidate:{ResultType void}{Text foo_10}{LeftParen (}{Placeholder int}{Comma , }{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC41: Completion contexts:
// CHECK-CC41-NEXT: Any type
// CHECK-CC41-NEXT: Any value
// CHECK-CC41-NEXT: Enum tag
// CHECK-CC41-NEXT: Union tag
// CHECK-CC41-NEXT: Struct tag
// CHECK-CC41-NEXT: Class name
// CHECK-CC41-NEXT: Nested name specifier
// CHECK-CC41-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:81:12 %s | FileCheck -check-prefix=CHECK-CC42 %s
// CHECK-CC42: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{CurrentParameter T}{Comma , }{Placeholder U}{RightParen )} (1)
// CHECK-CC42: Completion contexts:
// CHECK-CC42-NEXT: Any type
// CHECK-CC42-NEXT: Any value
// CHECK-CC42-NEXT: Enum tag
// CHECK-CC42-NEXT: Union tag
// CHECK-CC42-NEXT: Struct tag
// CHECK-CC42-NEXT: Class name
// CHECK-CC42-NEXT: Nested name specifier
// CHECK-CC42-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:81:15 %s | FileCheck -check-prefix=CHECK-CC43 %s
// CHECK-CC43: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter U}{RightParen )} (1)
// CHECK-CC43: Completion contexts:
// CHECK-CC43-NEXT: Any type
// CHECK-CC43-NEXT: Any value
// CHECK-CC43-NEXT: Enum tag
// CHECK-CC43-NEXT: Union tag
// CHECK-CC43-NEXT: Struct tag
// CHECK-CC43-NEXT: Class name
// CHECK-CC43-NEXT: Nested name specifier
// CHECK-CC43-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:82:17 %s | FileCheck -check-prefix=CHECK-CC44 %s
// CHECK-CC44: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder U}{RightParen )} (1)
// CHECK-CC44: Completion contexts:
// CHECK-CC44-NEXT: Any type
// CHECK-CC44-NEXT: Any value
// CHECK-CC44-NEXT: Enum tag
// CHECK-CC44-NEXT: Union tag
// CHECK-CC44-NEXT: Struct tag
// CHECK-CC44-NEXT: Class name
// CHECK-CC44-NEXT: Nested name specifier
// CHECK-CC44-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:82:20 %s | FileCheck -check-prefix=CHECK-CC45 %s
// CHECK-CC45: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter U}{RightParen )} (1)
// CHECK-CC45: Completion contexts:
// CHECK-CC45-NEXT: Any type
// CHECK-CC45-NEXT: Any value
// CHECK-CC45-NEXT: Enum tag
// CHECK-CC45-NEXT: Union tag
// CHECK-CC45-NEXT: Struct tag
// CHECK-CC45-NEXT: Class name
// CHECK-CC45-NEXT: Nested name specifier
// CHECK-CC45-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:83:28 %s | FileCheck -check-prefix=CHECK-CC46 %s
// CHECK-CC46: OverloadCandidate:{ResultType void}{Text foo_11}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC46: Completion contexts:
// CHECK-CC46-NEXT: Any type
// CHECK-CC46-NEXT: Any value
// CHECK-CC46-NEXT: Enum tag
// CHECK-CC46-NEXT: Union tag
// CHECK-CC46-NEXT: Struct tag
// CHECK-CC46-NEXT: Class name
// CHECK-CC46-NEXT: Nested name specifier
// CHECK-CC46-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:84:12 %s | FileCheck -check-prefix=CHECK-CC47 %s
// CHECK-CC47: OverloadCandidate:{ResultType void}{Text foo_12}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder int}{RightParen )} (1)
// CHECK-CC47: Completion contexts:
// CHECK-CC47-NEXT: Any type
// CHECK-CC47-NEXT: Any value
// CHECK-CC47-NEXT: Enum tag
// CHECK-CC47-NEXT: Union tag
// CHECK-CC47-NEXT: Struct tag
// CHECK-CC47-NEXT: Class name
// CHECK-CC47-NEXT: Nested name specifier
// CHECK-CC47-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:84:14 %s | FileCheck -check-prefix=CHECK-CC48 %s
// CHECK-CC48: OverloadCandidate:{ResultType void}{Text foo_12}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter void *}{RightParen )} (1)
// CHECK-CC48: Completion contexts:
// CHECK-CC48-NEXT: Any type
// CHECK-CC48-NEXT: Any value
// CHECK-CC48-NEXT: Enum tag
// CHECK-CC48-NEXT: Union tag
// CHECK-CC48-NEXT: Struct tag
// CHECK-CC48-NEXT: Class name
// CHECK-CC48-NEXT: Nested name specifier
// CHECK-CC48-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:85:12 %s | FileCheck -check-prefix=CHECK-CC49 %s
// CHECK-CC49: OverloadCandidate:{ResultType void}{Text foo_13}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC49: Completion contexts:
// CHECK-CC49-NEXT: Any type
// CHECK-CC49-NEXT: Any value
// CHECK-CC49-NEXT: Enum tag
// CHECK-CC49-NEXT: Union tag
// CHECK-CC49-NEXT: Struct tag
// CHECK-CC49-NEXT: Class name
// CHECK-CC49-NEXT: Nested name specifier
// CHECK-CC49-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:85:15 %s | FileCheck -check-prefix=CHECK-CC50 %s
// CHECK-CC50: OverloadCandidate:{ResultType void}{Text foo_13}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC50: Completion contexts:
// CHECK-CC50-NEXT: Any type
// CHECK-CC50-NEXT: Any value
// CHECK-CC50-NEXT: Enum tag
// CHECK-CC50-NEXT: Union tag
// CHECK-CC50-NEXT: Struct tag
// CHECK-CC50-NEXT: Class name
// CHECK-CC50-NEXT: Nested name specifier
// CHECK-CC50-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:85:19 %s | FileCheck -check-prefix=CHECK-CC51 %s
// CHECK-CC51: OverloadCandidate:{ResultType void}{Text foo_13}{LeftParen (}{Placeholder int}{Comma , }{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC51: Completion contexts:
// CHECK-CC51-NEXT: Any type
// CHECK-CC51-NEXT: Any value
// CHECK-CC51-NEXT: Enum tag
// CHECK-CC51-NEXT: Union tag
// CHECK-CC51-NEXT: Struct tag
// CHECK-CC51-NEXT: Class name
// CHECK-CC51-NEXT: Nested name specifier
// CHECK-CC51-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:87:12 %s | FileCheck -check-prefix=CHECK-CC52 %s
// CHECK-CC52: Completion contexts:
// CHECK-CC52-NEXT: Any type
// CHECK-CC52-NEXT: Any value
// CHECK-CC52-NEXT: Enum tag
// CHECK-CC52-NEXT: Union tag
// CHECK-CC52-NEXT: Struct tag
// CHECK-CC52-NEXT: Class name
// CHECK-CC52-NEXT: Nested name specifier
// CHECK-CC52-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:88:12 %s | FileCheck -check-prefix=CHECK-CC53 %s
// CHECK-CC53: Completion contexts:
// CHECK-CC53-NEXT: Any type
// CHECK-CC53-NEXT: Any value
// CHECK-CC53-NEXT: Enum tag
// CHECK-CC53-NEXT: Union tag
// CHECK-CC53-NEXT: Struct tag
// CHECK-CC53-NEXT: Class name
// CHECK-CC53-NEXT: Nested name specifier
// CHECK-CC53-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:89:17 %s | FileCheck -check-prefix=CHECK-CC54 %s
// CHECK-CC54: Completion contexts:
// CHECK-CC54-NEXT: Any type
// CHECK-CC54-NEXT: Any value
// CHECK-CC54-NEXT: Enum tag
// CHECK-CC54-NEXT: Union tag
// CHECK-CC54-NEXT: Struct tag
// CHECK-CC54-NEXT: Class name
// CHECK-CC54-NEXT: Nested name specifier
// CHECK-CC54-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:90:12 %s | FileCheck -check-prefix=CHECK-CC55 %s
// CHECK-CC55: Completion contexts:
// CHECK-CC55-NEXT: Any type
// CHECK-CC55-NEXT: Any value
// CHECK-CC55-NEXT: Enum tag
// CHECK-CC55-NEXT: Union tag
// CHECK-CC55-NEXT: Struct tag
// CHECK-CC55-NEXT: Class name
// CHECK-CC55-NEXT: Nested name specifier
// CHECK-CC55-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:91:14 %s | FileCheck -check-prefix=CHECK-CC56 %s
// CHECK-CC56: Completion contexts:
// CHECK-CC56-NEXT: Any type
// CHECK-CC56-NEXT: Any value
// CHECK-CC56-NEXT: Enum tag
// CHECK-CC56-NEXT: Union tag
// CHECK-CC56-NEXT: Struct tag
// CHECK-CC56-NEXT: Class name
// CHECK-CC56-NEXT: Nested name specifier
// CHECK-CC56-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:92:14 %s | FileCheck -check-prefix=CHECK-CC57 %s
// CHECK-CC57: Completion contexts:
// CHECK-CC57-NEXT: Any type
// CHECK-CC57-NEXT: Any value
// CHECK-CC57-NEXT: Enum tag
// CHECK-CC57-NEXT: Union tag
// CHECK-CC57-NEXT: Struct tag
// CHECK-CC57-NEXT: Class name
// CHECK-CC57-NEXT: Nested name specifier
// CHECK-CC57-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:93:19 %s | FileCheck -check-prefix=CHECK-CC58 %s
// CHECK-CC58: Completion contexts:
// CHECK-CC58-NEXT: Any type
// CHECK-CC58-NEXT: Any value
// CHECK-CC58-NEXT: Enum tag
// CHECK-CC58-NEXT: Union tag
// CHECK-CC58-NEXT: Struct tag
// CHECK-CC58-NEXT: Class name
// CHECK-CC58-NEXT: Nested name specifier
// CHECK-CC58-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:94:14 %s | FileCheck -check-prefix=CHECK-CC59 %s
// CHECK-CC59: Completion contexts:
// CHECK-CC59-NEXT: Any type
// CHECK-CC59-NEXT: Any value
// CHECK-CC59-NEXT: Enum tag
// CHECK-CC59-NEXT: Union tag
// CHECK-CC59-NEXT: Struct tag
// CHECK-CC59-NEXT: Class name
// CHECK-CC59-NEXT: Nested name specifier
// CHECK-CC59-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:104:14 %s | FileCheck -check-prefix=CHECK-CC60 %s
// CHECK-CC60: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC60: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC60: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC60: Completion contexts:
// CHECK-CC60-NEXT: Any type
// CHECK-CC60-NEXT: Any value
// CHECK-CC60-NEXT: Enum tag
// CHECK-CC60-NEXT: Union tag
// CHECK-CC60-NEXT: Struct tag
// CHECK-CC60-NEXT: Class name
// CHECK-CC60-NEXT: Nested name specifier
// CHECK-CC60-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:106:11 %s | FileCheck -check-prefix=CHECK-CC61 %s
// CHECK-CC61: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC61: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC61: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC61: Completion contexts:
// CHECK-CC61-NEXT: Any type
// CHECK-CC61-NEXT: Any value
// CHECK-CC61-NEXT: Enum tag
// CHECK-CC61-NEXT: Union tag
// CHECK-CC61-NEXT: Struct tag
// CHECK-CC61-NEXT: Class name
// CHECK-CC61-NEXT: Nested name specifier
// CHECK-CC61-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:111:16 %s | FileCheck -check-prefix=CHECK-CC62 %s
// CHECK-CC62: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC62: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC62: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC62: Completion contexts:
// CHECK-CC62-NEXT: Any type
// CHECK-CC62-NEXT: Any value
// CHECK-CC62-NEXT: Enum tag
// CHECK-CC62-NEXT: Union tag
// CHECK-CC62-NEXT: Struct tag
// CHECK-CC62-NEXT: Class name
// CHECK-CC62-NEXT: Nested name specifier
// CHECK-CC62-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:131:23 %s | FileCheck -check-prefix=CHECK-CC63 %s
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T1 a}{Comma , }{Placeholder T2 b}{RightParen )} (1)
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T1 a}{Comma , }{Placeholder T2 b}{Comma , }{Placeholder float c}{RightParen )} (1)
// CHECK-CC63: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T2 a}{Comma , }{Placeholder int b}{Comma , }{Placeholder T1 c}{RightParen )} (1)

// RUN: c-index-test -code-completion-at=%s:133:11 %s | FileCheck -check-prefix=CHECK-CC64 %s
// CHECK-CC64: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC64: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC64: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC64: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T1 a}{Comma , }{Placeholder T2 b}{RightParen )} (1)
// CHECK-CC64: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T2 a}{Comma , }{Placeholder int b}{Comma , }{Placeholder T1 c}{RightParen )} (1)

// RUN: c-index-test -code-completion-at=%s:138:25 %s | FileCheck -check-prefix=CHECK-CC65 %s
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{RightParen )} (1)
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter float}{RightParen )} (1)
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T1 a}{Comma , }{Placeholder T2 b}{RightParen )} (1)
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T1 a}{Comma , }{Placeholder T2 b}{Comma , }{Placeholder float c}{RightParen )} (1)
// CHECK-CC65: OverloadCandidate:{ResultType void}{Text foo_1}{LeftParen (}{CurrentParameter T2 a}{Comma , }{Placeholder int b}{Comma , }{Placeholder T1 c}{RightParen )} (1)

