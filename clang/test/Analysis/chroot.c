// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Chroot -analyzer-output=text -verify %s

extern int chroot(const char* path);
extern int chdir(const char* path);

void foo(void) {
}

void f1(void) {
  chroot("/usr/local"); // expected-note {{chroot called here}}
  foo();
  // expected-warning@-1 {{No call of chdir("/") immediately after chroot}}
  // expected-note@-2    {{No call of chdir("/") immediately after chroot}}
}

void f2(void) {
  chroot("/usr/local"); // root changed.
  chdir("/"); // enter the jail.
  foo(); // no-warning
}

void f3(void) {
  chroot("/usr/local"); // expected-note {{chroot called here}}
  chdir("../"); // change working directory, still out of jail.
  foo();
  // expected-warning@-1 {{No call of chdir("/") immediately after chroot}}
  // expected-note@-2    {{No call of chdir("/") immediately after chroot}}
}

void f4(void) {
  if (chroot("/usr/local") == 0) {
      chdir("../"); // change working directory, still out of jail.
  }
}

void f5(void) {
  int v = chroot("/usr/local");
  if (v == -1) {
      foo();        // no warning, chroot failed
      chdir("../"); // change working directory, still out of jail.
  }
}

void f6(void) {
  if (chroot("/usr/local") == -1) {
      chdir("../"); // change working directory, still out of jail.
  }
}

void f7(void) {
  int v = chroot("/usr/local"); // expected-note {{chroot called here}}
  if (v == -1) { // expected-note {{Taking false branch}}
      foo();        // no warning, chroot failed
      chdir("../"); // change working directory, still out of jail.
  } else {
      foo();
      // expected-warning@-1 {{No call of chdir("/") immediately after chroot}}
      // expected-note@-2    {{No call of chdir("/") immediately after chroot}}
  }
}

void f8() {
  chroot("/usr/local"); // expected-note {{chroot called here}}
  chdir("/usr"); // This chdir was ineffective because it's not exactly `chdir("/")`.
  foo();
  // expected-warning@-1 {{No call of chdir("/") immediately after chroot}}
  // expected-note@-2    {{No call of chdir("/") immediately after chroot}}
}
