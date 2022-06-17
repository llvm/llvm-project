// RUN: %check_clang_tidy %s bugprone-dynamic-static-initializers %t -- -- -fno-threadsafe-statics

int fact(int n) {
  return (n == 0) ? 1 : n * fact(n - 1);
}

int static_thing = fact(5);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: static variable 'static_thing' may be dynamically initialized in this header file [bugprone-dynamic-static-initializers]

int sample() {
    int x;
    return x;
}

int dynamic_thing = sample();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: static variable 'dynamic_thing' may be dynamically initialized in this header file [bugprone-dynamic-static-initializers]

int not_so_bad = 12 + 4942; // no warning

extern int bar();

int foo() {
  static int k = bar();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: static variable 'k' may be dynamically initialized in this header file [bugprone-dynamic-static-initializers]
  return k;
}

int bar2() {
  return 7;
}

int foo2() {
  // This may work fine when optimization is enabled because bar() can
  // be turned into a constant 7.  But without optimization, it can
  // cause problems. Therefore, we must err on the side of conservatism.
  static int x = bar2();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: static variable 'x' may be dynamically initialized in this header file [bugprone-dynamic-static-initializers]
  return x;
}

int foo3() {
  static int p = 7 + 83; // no warning
  return p;
}
