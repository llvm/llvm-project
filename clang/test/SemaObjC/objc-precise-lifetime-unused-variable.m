// RUN: %clang_cc1 -triple x86_64-apple-macos11 -fsyntax-only -fobjc-arc -fblocks -verify -Wunused-but-set-variable -Wno-objc-root-class %s

id getFoo(void);

void test() {
  // no diagnostics for objects with precise lifetime semantics.
  __attribute__((objc_precise_lifetime)) id x;
  x = getFoo();

  id x2; // expected-warning {{variable 'x2' set but not used}}
  x2 = getFoo();

  do {
    __attribute__((objc_precise_lifetime)) id y;
    y = getFoo();

    id y2; // expected-warning {{variable 'y2' set but not used}}
    y2 = getFoo();
  } while(0);

  x = ((void *)0);
}
