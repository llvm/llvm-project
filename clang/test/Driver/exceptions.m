// RUN: %clang -target x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -fno-exceptions %s

void f1(void) {
  @throw @"A";
}

void f0(void) {
  @try {
    f1();
  } @catch (id x) {
    ;
  }
}

int main(void) {
  f0();
  return 0;
}
