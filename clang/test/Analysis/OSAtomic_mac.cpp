// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -analyzer-checker=core,osx -verify -fblocks   %s
// expected-no-diagnostics

// Test handling of OSAtomicCompareAndSwap when C++ inserts "no-op" casts and we
// do a forced load and binding to the environment on an expression that would regularly
// not have an environment binding.  This previously triggered a crash.
// NOTE: It is critical that the function called is OSAtomicCompareAndSwapIntBarrier.
bool OSAtomicCompareAndSwapIntBarrier( int __oldValue, int __newValue, volatile int *__theValue ) ;
static int _rdar9339920_x = 0;
int rdar9339920_aux();

int rdar9339920_test() {
  int rdar9339920_x = rdar9339920_aux();
  if (rdar9339920_x != _rdar9339920_x) {
    if (OSAtomicCompareAndSwapIntBarrier(_rdar9339920_x, rdar9339920_x, &_rdar9339920_x))
      return 1;
  }
  return 0;
}

