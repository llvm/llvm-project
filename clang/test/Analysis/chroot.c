// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.unix.Chroot -verify %s

extern int chroot(const char* path);
extern int chdir(const char* path);

void foo(void) {
}

void f1(void) {
  chroot("/usr/local"); // root changed.
  foo(); // expected-warning {{No call of chdir("/") immediately after chroot}}
}

void f2(void) {
  chroot("/usr/local"); // root changed.
  chdir("/"); // enter the jail.
  foo(); // no-warning
}

void f3(void) {
  chroot("/usr/local"); // root changed.
  chdir("../"); // change working directory, still out of jail.
  foo(); // expected-warning {{No call of chdir("/") immediately after chroot}}
}
