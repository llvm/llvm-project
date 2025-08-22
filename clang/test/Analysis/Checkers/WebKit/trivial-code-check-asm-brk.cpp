// RUN: %clang_analyze_cc1 -triple arm-darwin -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

void crash()
{
  __asm__ volatile ("brk #0xc471");
  __builtin_unreachable();
}

class SomeObj {
public:
  void ref();
  void deref();

  void someWork() { crash(); }
};

SomeObj* provide();

void doSomeWork() {
  provide()->someWork();
}
