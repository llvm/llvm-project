// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 \
// RUN:   -fsanitize=kcfi -Wno-deprecated-non-prototype -emit-llvm -o /dev/null \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 \
// RUN:   -fsanitize=kcfi -Wno-deprecated-non-prototype \
// RUN:   -Wno-kcfi-unprototyped-call -emit-llvm -o /dev/null -verify=none %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 \
// RUN:   -Wno-deprecated-non-prototype -emit-llvm -o /dev/null -verify=none %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 \
// RUN:   -fsanitize=kcfi -emit-llvm -o /dev/null -verify=none -DC23 %s

/// -fsanitize=kcfi doesn't check indirect calls through a function type without
/// a prototype.

// none-no-diagnostics

int knr();
int proto(int);
int impl(int x) { return x; }
int alias(int) __attribute__((alias("impl")));

typedef int unprototyped_fn();

struct ops {
  int (*unprototyped)();
  int (*prototyped)(int);
};

int calls(int (*unprototyped)(), int (*prototyped)(int), struct ops *o,
          unprototyped_fn *typedef_fn) {
  int r = 0;
#ifdef C23
  r += unprototyped();
  r += o->unprototyped();
  r += typedef_fn();
#else
  r += unprototyped(1);    // expected-warning {{indirect call through a function type without a prototype is not checked by '-fsanitize=kcfi'}}
  r += unprototyped();     // expected-warning {{indirect call through a function type without a prototype is not checked by '-fsanitize=kcfi'}}
  r += (*unprototyped)(1); // expected-warning {{indirect call through a function type without a prototype is not checked by '-fsanitize=kcfi'}}
  r += o->unprototyped(1); // expected-warning {{indirect call through a function type without a prototype is not checked by '-fsanitize=kcfi'}}
  r += typedef_fn(1);      // expected-warning {{indirect call through a function type without a prototype is not checked by '-fsanitize=kcfi'}}
  r += knr(1);
  r += (*knr)(1);
  r += ((int (*)())alias)(1);
  r += ((int (*)())impl)(1);
#endif
  r += prototyped(1);
  r += o->prototyped(1);
  r += ((int (*)(int))unprototyped)(1);
  r += proto(1);
  return r;
}

__attribute__((no_sanitize("kcfi"))) int no_kcfi(int (*unprototyped)()) {
#ifdef C23
  return unprototyped();
#else
  return unprototyped(1);
#endif
}
