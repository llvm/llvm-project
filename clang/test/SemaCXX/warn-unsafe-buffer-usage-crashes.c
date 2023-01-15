// RUN: %clang_cc1 -Wunsafe-buffer-usage %s -verify %s

void gnu_stmtexpr_crash(void) {
  struct A {};
  struct B {
    struct A a;
  };

  struct B b = {{
    // This is a statement-expression (GNU extension).
    ({ int x; }) // no-crash // expected-warning{{excess elements in struct initializer}}
  }};
}
