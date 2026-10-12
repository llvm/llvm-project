// RUN: %clang_cc1 -fsyntax-only -verify -Wunreachable-code -Wreturn-type -Wimplicit-fallthrough %s

// 'analyzer_noreturn' calls can return, so the code after them is not
// unreachable. -Wreturn-type and -Wimplicit-fallthrough do treat them as
// noreturn.

void analyzer_stop(const char *) __attribute__((analyzer_noreturn));
void real_stop(void) __attribute__((noreturn));
void use(int);

void after_analyzer_noreturn(int x) {
  if (x)
    return;
  analyzer_stop("bad");
  use(1); // no warning
}

void analyzer_then_real(void) {
  analyzer_stop("bad");
  real_stop();
  use(1); // expected-warning {{code will never be executed}}
}

int return_type(int x) {
  if (x)
    return 1;
  analyzer_stop("bad");
} // no warning

void fallthrough(int x) {
  switch (x) {
  case 1:
    analyzer_stop("bad");
  case 2: // no warning
    use(2);
    break;
  }
}
