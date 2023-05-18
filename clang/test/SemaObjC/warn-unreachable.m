// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code-aggressive -Wno-unused-value -Wno-covered-switch-default

// This previously triggered a warning from -Wunreachable-code because of
// a busted CFG.
typedef signed char BOOL;
BOOL radar10989084(void) {
  @autoreleasepool {  // no-warning
    return __objc_yes;
  }
}

// Test the warning works.
void test_unreachable(void) {
  return;
  return; // expected-warning {{will never be executed}}
}

#define NO __objc_no
#define YES __objc_yes
#define CONFIG NO

// Test that 'NO' and 'YES' are not treated as configuration macros.
int test_NO(void) {
  if (NO)
    return 1; // expected-warning {{will never be executed}}
  else
    return 0;
}

int test_YES(void) {
  if (YES)
    return 1;
  else
    return 0; // expected-warning {{will never be executed}}
}

int test_CONFIG(void) {
  if (CONFIG)
    return 1;
  else
    return 0;
}

// FIXME: This should at some point report a warning
// that the loop increment is unreachable.
void test_loop_increment(id container) {
  for (id x in container) { // no-warning
    break;
  }
}

void calledFun(void) {}

// Test "silencing" with parentheses.
void test_with_paren_silencing(int x) {
  if (NO) calledFun(); // expected-warning {{will never be executed}} expected-note {{silence by adding parentheses to mark code as explicitly dead}}
  if ((NO)) calledFun(); // no-warning

  if (YES) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun();
  else
    calledFun(); // expected-warning {{will never be executed}}

  if ((YES))
    calledFun();
  else
    calledFun(); // no-warning
  
  if (!YES) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun(); // expected-warning {{code will never be executed}}
  else
    calledFun();
  
  if ((!YES))
    calledFun(); // no-warning
  else
    calledFun();
  
  if (!(YES))
    calledFun(); // no-warning
  else
    calledFun();
}

