// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin -std=c++11 -fblocks -Wframe-larger-than=70 -Wno-stdlibcxx-not-found -Xclang -verify -o /dev/null -c %s
// RUN: %clang -target i386-apple-darwin -std=c++11 -fblocks -Wframe-larger-than=70 -Wno-stdlibcxx-not-found -Xclang -verify -o /dev/null -c %s -DIS_SYSHEADER

// Test that:
//  * The driver passes the option through to the backend.
//  * The frontend diagnostic handler 'demangles' and resolves the correct function definition.

// Test that link invocations don't emit an "argument unused during compilation" diagnostic.
// RUN: touch %t.o
// RUN: %clang -Werror -Wno-msvc-not-found -Wno-liblto -Wframe-larger-than=0 %t.o -###  2>&1 | not grep ' error: '

// TODO: Support rich backend diagnostics for Objective-C methods.

// Backend diagnostics aren't suppressed in system headers because such results
// are significant and actionable.
#ifdef IS_HEADER

#ifdef IS_SYSHEADER
#pragma clang system_header
#endif

extern void doIt(char *);

void frameSizeWarning(int, int) {}

void frameSizeWarning();

void frameSizeWarning() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in function 'frameSizeWarning'}}
  char buffer[80];
  doIt(buffer);
}

void frameSizeWarning();

void frameSizeWarning(int) {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wframe-larger-than="
void frameSizeWarningIgnored() {
  char buffer[80];
  doIt(buffer);
}
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#ifndef IS_SYSHEADER
// expected-warning@+2 {{unknown warning group '-Wframe-larger-than'}}
#endif
#pragma GCC diagnostic ignored "-Wframe-larger-than"
#pragma GCC diagnostic pop

void frameSizeLocalClassWarning() {
  struct S {
    S() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in function 'frameSizeLocalClassWarning()::S::S'}}
      char buffer[80];
      doIt(buffer);
    }
  };
  S();
}

void frameSizeLambdaWarning() {
  auto fn =
      []() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in lambda expression}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}

void frameSizeBlocksWarning() {
  auto fn =
      ^() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in block literal}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}

#else

#define IS_HEADER
#include __FILE__
#endif
