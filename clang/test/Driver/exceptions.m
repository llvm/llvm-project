// RUN: %clang -target x86_64-apple-darwin9 \
// RUN:   -fsyntax-only -fno-exceptions %s

void f1() {
  @throw @"A";
}

void f0() {
  @try {
    f1();
  } @catch (id x) {
    ;
  }
}

int main() {
  f0();
  return 0;
}
