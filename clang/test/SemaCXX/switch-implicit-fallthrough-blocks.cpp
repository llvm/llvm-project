// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=c++11 -Wimplicit-fallthrough %s

void fallthrough_in_blocks() {
  void (^block)() = ^{
    int x = 0;
    switch (x) {
    case 0:
      x++;
      [[clang::fallthrough]]; // no diagnostics
    case 1:
      x++;
    default: // \
        expected-warning{{unannotated fall-through between switch labels}} \
        expected-note{{insert 'break;' to avoid fall-through}}
      break;
    }
  };
  block();
}
