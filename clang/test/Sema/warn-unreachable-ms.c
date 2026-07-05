// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fsyntax-only -verify -fms-extensions -Wunreachable-code

void f(void);

void g1(void) {
  __try {
    f();
    __leave;
    f();  // expected-warning{{will never be executed}}
  } __except(1) {
    f();
  }

  // Completely empty.
  __try {
  } __except(1) {
  }

  __try {
    f();
    return;
  } __except(1) {  // Filter expression should not be marked as unreachable.
    // Empty __except body.
  }
}

void g2(void) {
  __try {
    // Nested __try.
    __try {
      f();
      __leave;
      f(); // expected-warning{{will never be executed}}
    } __except(2) {
    }
    f();
    __leave;
    f(); // expected-warning{{will never be executed}}
  } __except(1) {
    f();
  }
}

void g3(void) {
  __try {
    __try {
      f();
    } __except (1) {
      __leave; // should exit outer try
    }
    __leave;
    f(); // expected-warning{{never be executed}}
  } __except (1) {
  }
}
