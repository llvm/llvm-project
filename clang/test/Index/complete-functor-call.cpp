// Note: the run lines follow their respective tests, since line/column
// matter in this test.

template<class V>
struct S {
  void operator()(int) const {}
  template<class T> void operator()(T) const {}
  template<class T> void operator()(V, T, T) const {}
  template<class T> const S<T> *operator()(const S<T> &s) const { return &s; }
};

void foo(S<void *> &s) { s(42); }

int main() {
  S<void *> s;
  s(42);
  s(s);
  s(0, s, s);
  (*S<void *>()(S<int>()))(42, 42, 42);

  s(42,);
  s(s,);
  s(0, 42, 42,);
}

// RUN: c-index-test -code-completion-at=%s:16:5 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC1: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC1: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Any type
// CHECK-CC1-NEXT: Any value
// CHECK-CC1-NEXT: Enum tag
// CHECK-CC1-NEXT: Union tag
// CHECK-CC1-NEXT: Struct tag
// CHECK-CC1-NEXT: Class name
// CHECK-CC1-NEXT: Nested name specifier
// CHECK-CC1-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:17:5 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Any type
// CHECK-CC2-NEXT: Any value
// CHECK-CC2-NEXT: Enum tag
// CHECK-CC2-NEXT: Union tag
// CHECK-CC2-NEXT: Struct tag
// CHECK-CC2-NEXT: Class name
// CHECK-CC2-NEXT: Nested name specifier
// CHECK-CC2-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:18:5 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC3: Completion contexts:
// CHECK-CC3-NEXT: Any type
// CHECK-CC3-NEXT: Any value
// CHECK-CC3-NEXT: Enum tag
// CHECK-CC3-NEXT: Union tag
// CHECK-CC3-NEXT: Struct tag
// CHECK-CC3-NEXT: Class name
// CHECK-CC3-NEXT: Nested name specifier
// CHECK-CC3-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:18:7 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{Placeholder void *}{Comma , }{CurrentParameter T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC4: Completion contexts:
// CHECK-CC4-NEXT: Any type
// CHECK-CC4-NEXT: Any value
// CHECK-CC4-NEXT: Enum tag
// CHECK-CC4-NEXT: Union tag
// CHECK-CC4-NEXT: Struct tag
// CHECK-CC4-NEXT: Class name
// CHECK-CC4-NEXT: Nested name specifier
// CHECK-CC4-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:18:10 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{Placeholder void *}{Comma , }{Placeholder S<void *>}{Comma , }{CurrentParameter S<void *>}{RightParen )} (1)
// CHECK-CC5: Completion contexts:
// CHECK-CC5-NEXT: Any type
// CHECK-CC5-NEXT: Any value
// CHECK-CC5-NEXT: Enum tag
// CHECK-CC5-NEXT: Union tag
// CHECK-CC5-NEXT: Struct tag
// CHECK-CC5-NEXT: Class name
// CHECK-CC5-NEXT: Nested name specifier
// CHECK-CC5-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:19:17 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC6: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC6: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC6: OverloadCandidate:{ResultType const S<T> *}{Text operator()}{LeftParen (}{CurrentParameter const S<T> &s}{RightParen )} (1)
// CHECK-CC6: Completion contexts:
// CHECK-CC6-NEXT: Any type
// CHECK-CC6-NEXT: Any value
// CHECK-CC6-NEXT: Enum tag
// CHECK-CC6-NEXT: Union tag
// CHECK-CC6-NEXT: Struct tag
// CHECK-CC6-NEXT: Class name
// CHECK-CC6-NEXT: Nested name specifier
// CHECK-CC6-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:19:28 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC7: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC7: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC7: OverloadCandidate:{ResultType const S<T> *}{Text operator()}{LeftParen (}{CurrentParameter const S<T> &s}{RightParen )} (1)
// CHECK-CC7: Completion contexts:
// CHECK-CC7-NEXT: Any type
// CHECK-CC7-NEXT: Any value
// CHECK-CC7-NEXT: Enum tag
// CHECK-CC7-NEXT: Union tag
// CHECK-CC7-NEXT: Struct tag
// CHECK-CC7-NEXT: Class name
// CHECK-CC7-NEXT: Nested name specifier
// CHECK-CC7-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:19:31 %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{Placeholder int}{Comma , }{CurrentParameter T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC8: Completion contexts:
// CHECK-CC8-NEXT: Any type
// CHECK-CC8-NEXT: Any value
// CHECK-CC8-NEXT: Enum tag
// CHECK-CC8-NEXT: Union tag
// CHECK-CC8-NEXT: Struct tag
// CHECK-CC8-NEXT: Class name
// CHECK-CC8-NEXT: Nested name specifier
// CHECK-CC8-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:19:35 %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{Placeholder int}{Comma , }{Placeholder int}{Comma , }{CurrentParameter int}{RightParen )} (1)
// CHECK-CC9: Completion contexts:
// CHECK-CC9-NEXT: Any type
// CHECK-CC9-NEXT: Any value
// CHECK-CC9-NEXT: Enum tag
// CHECK-CC9-NEXT: Union tag
// CHECK-CC9-NEXT: Struct tag
// CHECK-CC9-NEXT: Class name
// CHECK-CC9-NEXT: Nested name specifier
// CHECK-CC9-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:21:8 %s | FileCheck -check-prefix=CHECK-CC10 %s
// CHECK-CC10: Completion contexts:
// CHECK-CC10-NEXT: Any type
// CHECK-CC10-NEXT: Any value
// CHECK-CC10-NEXT: Enum tag
// CHECK-CC10-NEXT: Union tag
// CHECK-CC10-NEXT: Struct tag
// CHECK-CC10-NEXT: Class name
// CHECK-CC10-NEXT: Nested name specifier
// CHECK-CC10-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:22:7 %s | FileCheck -check-prefix=CHECK-CC11 %s
// CHECK-CC11: Completion contexts:
// CHECK-CC11-NEXT: Any type
// CHECK-CC11-NEXT: Any value
// CHECK-CC11-NEXT: Enum tag
// CHECK-CC11-NEXT: Union tag
// CHECK-CC11-NEXT: Struct tag
// CHECK-CC11-NEXT: Class name
// CHECK-CC11-NEXT: Nested name specifier
// CHECK-CC11-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:23:15 %s | FileCheck -check-prefix=CHECK-CC12 %s
// CHECK-CC12: Completion contexts:
// CHECK-CC12-NEXT: Any type
// CHECK-CC12-NEXT: Any value
// CHECK-CC12-NEXT: Enum tag
// CHECK-CC12-NEXT: Union tag
// CHECK-CC12-NEXT: Struct tag
// CHECK-CC12-NEXT: Class name
// CHECK-CC12-NEXT: Nested name specifier
// CHECK-CC12-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:12:28 %s | FileCheck -check-prefix=CHECK-CC13 %s
// CHECK-CC13: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC13: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter T}{RightParen )} (1)
// CHECK-CC13: OverloadCandidate:{ResultType void}{Text operator()}{LeftParen (}{CurrentParameter void *}{Comma , }{Placeholder T}{Comma , }{Placeholder T}{RightParen )} (1)
// CHECK-CC13: Completion contexts:
// CHECK-CC13-NEXT: Any type
// CHECK-CC13-NEXT: Any value
// CHECK-CC13-NEXT: Enum tag
// CHECK-CC13-NEXT: Union tag
// CHECK-CC13-NEXT: Struct tag
// CHECK-CC13-NEXT: Class name
// CHECK-CC13-NEXT: Nested name specifier
// CHECK-CC13-NEXT: Objective-C interface
