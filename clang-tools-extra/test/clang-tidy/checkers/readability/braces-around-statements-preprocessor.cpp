// RUN: %check_clang_tidy -check-suffix=,UNDEFINED %s readability-braces-around-statements %t-undefined -- -- -UFOO
// RUN: %clang_cc1 -fsyntax-only -UFOO %t-undefined.cpp
// RUN: %clang_cc1 -fsyntax-only -DFOO=1 %t-undefined.cpp
// RUN: %check_clang_tidy -check-suffix=,DEFINED %s google-readability-braces-around-statements %t-defined -- -- -DFOO=1
// RUN: %clang_cc1 -fsyntax-only -UFOO %t-defined.cpp
// RUN: %clang_cc1 -fsyntax-only -DFOO=1 %t-defined.cpp

void foo();
void bar();
void consume(int);

// The opening brace would be outside #ifdef and the closing brace inside it.
void conditional_body(int i) {
  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i > 0)
#ifdef FOO
    foo();
#else
    bar();
#endif
}
// CHECK-FIXES-LABEL: void conditional_body(int i) {
// CHECK-FIXES: if (i > 0){{$}}
// CHECK-FIXES-NEXT: #ifdef FOO
// CHECK-FIXES-NEXT: foo();
// CHECK-FIXES-NEXT: #else
// CHECK-FIXES-NEXT: bar();
// CHECK-FIXES-NEXT: #endif
// CHECK-FIXES-NEXT: }

// The opening brace would be inside #if and the closing brace outside it.
void conditional_header(int i) {
#if FOO
  // CHECK-MESSAGES-DEFINED: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i > 0)
#else
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i < 0)
#endif
    foo();
}
// CHECK-FIXES-LABEL: void conditional_header(int i) {
// CHECK-FIXES: if (i > 0){{$}}
// CHECK-FIXES: if (i < 0){{$}}
// CHECK-FIXES-NEXT: #endif
// CHECK-FIXES-NEXT: foo();
// CHECK-FIXES-NEXT: }

// The same protection applies to loops.
void conditional_loop(int i) {
  // CHECK-MESSAGES: :[[@LINE+1]]:16: warning: statement should be inside braces
  while (i > 0)
#ifdef FOO
    foo();
#else
    bar();
#endif
}
// CHECK-FIXES-LABEL: void conditional_loop(int i) {
// CHECK-FIXES: while (i > 0){{$}}
// CHECK-FIXES-NEXT: #ifdef FOO
// CHECK-FIXES-NEXT: foo();
// CHECK-FIXES-NEXT: #else
// CHECK-FIXES-NEXT: bar();
// CHECK-FIXES-NEXT: #endif
// CHECK-FIXES-NEXT: }

// An explicit else location must not bypass the region check.
void conditional_else(int i) {
  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i > 0)
    foo();
#ifdef FOO
  else {
    bar();
  }
#endif
}
// CHECK-FIXES-LABEL: void conditional_else(int i) {
// CHECK-FIXES-DEFINED: if (i > 0){{$}}
// CHECK-FIXES-UNDEFINED: if (i > 0) {
// CHECK-FIXES-NEXT: foo();
// CHECK-FIXES-UNDEFINED-NEXT: }
// CHECK-FIXES-NEXT: #ifdef FOO
// CHECK-FIXES-NEXT: else {
// CHECK-FIXES-NEXT: bar();
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: #endif
// CHECK-FIXES-NEXT: }

// Braces within one conditional region remain safe.
void same_region(int i) {
#ifdef FOO
  // CHECK-MESSAGES-DEFINED: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i > 0)
    foo();
#else
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i < 0)
    bar();
#endif
}
// CHECK-FIXES-LABEL: void same_region(int i) {
// CHECK-FIXES-DEFINED: if (i > 0) {
// CHECK-FIXES-DEFINED-NEXT: foo();
// CHECK-FIXES-DEFINED-NEXT: }
// CHECK-FIXES-DEFINED-NEXT: #else
// CHECK-FIXES-UNDEFINED: if (i < 0) {
// CHECK-FIXES-UNDEFINED-NEXT: bar();
// CHECK-FIXES-UNDEFINED-NEXT: }
// CHECK-FIXES-UNDEFINED-NEXT: #endif

// A complete conditional block inside the new braces remains safe.
void enclosed_region(int i) {
  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: statement should be inside braces
  if (i > 0)
    consume(
#ifdef FOO
        1
#else
        2
#endif
    );
}
// CHECK-FIXES-LABEL: void enclosed_region(int i) {
// CHECK-FIXES: if (i > 0) {
// CHECK-FIXES-NEXT: consume(
// CHECK-FIXES-NEXT: #ifdef FOO
// CHECK-FIXES-NEXT: 1
// CHECK-FIXES-NEXT: #else
// CHECK-FIXES-NEXT: 2
// CHECK-FIXES-NEXT: #endif
// CHECK-FIXES-NEXT: );
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
