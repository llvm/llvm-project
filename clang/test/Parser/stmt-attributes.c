// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_extension(statement_attributes_with_gnu_syntax)
#error "We should have statement attributes with GNU syntax support"
#endif

void foo(int i) {

  __attribute__((unknown_attribute)); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  __attribute__(()) {}
  __attribute__(()) if (0) {}
  __attribute__(()) for (;;);
  __attribute__(()) do {
    __attribute__(()) continue;
  }
  while (0)
    ;
  __attribute__(()) while (0);

  __attribute__(()) switch (i) {
    __attribute__(()) case 0 :
    __attribute__(()) default :
    __attribute__(()) break;
  }

  __attribute__(()) goto here;
  __attribute__(()) here :

  __attribute__(()) return;

  __attribute__((noreturn)) {}             // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  __attribute__((noreturn)) if (0) {}      // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  __attribute__((noreturn)) for (;;);      // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  __attribute__((noreturn)) do {           // expected-error {{'noreturn' attribute cannot be applied to a statement}}
    __attribute__((unavailable)) continue; // expected-error {{'unavailable' attribute cannot be applied to a statement}}
  }
  while (0)
    ;
  __attribute__((unknown_attribute)) while (0); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}

  __attribute__((unused)) switch (i) {         // expected-error {{'unused' attribute cannot be applied to a statement}}
  __attribute__((uuid)) case 0:                // expected-warning {{unknown attribute 'uuid' ignored}}
  __attribute__((visibility(""))) default:         // expected-error {{'visibility' attribute cannot be applied to a statement}}
    __attribute__((nonnull)) break; // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  }

  __attribute__((fastcall)) goto there; // expected-error {{'fastcall' attribute cannot be applied to a statement}}
  __attribute__((noinline)) there :     // expected-warning {{'noinline' attribute only applies to functions and statements}}

                                    __attribute__((weakref)) return; // expected-error {{'weakref' attribute only applies to variables and functions}}

  __attribute__((nonnull));            // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  __attribute__((nonnull)) {}          // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  __attribute__((nonnull)) if (0) {}   // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  __attribute__((nonnull)) for (;;);   // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  __attribute__((nonnull)) do {        // expected-error {{'nonnull' attribute cannot be applied to a statement}}
    __attribute__((nonnull)) continue; // expected-error {{'nonnull' attribute cannot be applied to a statement}} ignored}}
  }
  while (0)
    ;
  __attribute__((nonnull)) while (0); // expected-error {{'nonnull' attribute cannot be applied to a statement}}

  __attribute__((nonnull)) switch (i) { // expected-error {{'nonnull' attribute cannot be applied to a statement}} ignored}}
  __attribute__((nonnull)) case 0:      // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  __attribute__((nonnull)) default:     // expected-error {{'nonnull' attribute cannot be applied to a statement}}
    __attribute__((nonnull)) break;     // expected-error {{'nonnull' attribute cannot be applied to a statement}}
  }

  __attribute__((nonnull)) goto here; // expected-error {{'nonnull' attribute cannot be applied to a statement}}

  __attribute__((nonnull)) return; // expected-error {{'nonnull' attribute cannot be applied to a statement}}
}

void bar(void);

void foobar(void) {
  __attribute__((nomerge)) bar();
  __attribute__(()) bar();                // expected-error {{expected identifier or '('}}
  __attribute__((unused, nomerge)) bar(); // expected-error {{expected identifier or '('}}
  __attribute__((nomerge, unused)) bar(); // expected-error {{expected identifier or '('}}
  __attribute__((nomerge(1, 2))) bar();   // expected-error {{'nomerge' attribute takes no arguments}}
  int x;
  __attribute__((nomerge)) x = 10; // expected-warning {{'nomerge' attribute is ignored because there exists no call expression inside the statement}}

  __attribute__((nomerge)) label : bar(); // expected-error {{'nomerge' attribute only applies to functions, statements and variables}}
}

int f(void);

__attribute__((nomerge)) static int (*fptr)(void);
__attribute__((nomerge)) static int i; // expected-warning {{'nomerge' attribute is ignored because 'i' is not a function pointer}}
struct buz {} __attribute__((nomerge)); // expected-error {{'nomerge' attribute only applies to functions, statements and variables}}
