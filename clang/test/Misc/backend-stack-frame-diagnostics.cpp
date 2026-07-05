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

void frameSizeWarning() { // expected-warning-re {{stack frame size ({{[0-9]+}}) exceeds limit ({{[0-9]+}}) in 'frameSizeWarning()'}}
  char buffer[80];
  doIt(buffer);
}

void frameSizeWarning();

void frameSizeWarning(int) {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wframe-larger-than"
void frameSizeWarningIgnored() {
  char buffer[80];
  doIt(buffer);
}
#pragma GCC diagnostic pop

void frameSizeLocalClassWarning() {
  struct S {
    S() { // expected-warning-re {{stack frame size ({{[0-9]+}}) exceeds limit ({{[0-9]+}}) in 'frameSizeLocalClassWarning()::S::S()'}}
      char buffer[80];
      doIt(buffer);
    }
  };
  S();
}

void frameSizeLambdaWarning() {
  auto fn =
      []() { // expected-warning-re {{stack frame size ({{[0-9]+}}) exceeds limit ({{[0-9]+}}) in 'frameSizeLambdaWarning()::$_0::operator()() const'}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}

void frameSizeBlocksWarning() {
  auto fn =
      ^() { // expected-warning-re {{stack frame size ({{[0-9]+}}) exceeds limit ({{[0-9]+}}) in 'invocation function for block in frameSizeBlocksWarning()'}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}

#else

#define IS_HEADER
#include __FILE__
#endif
