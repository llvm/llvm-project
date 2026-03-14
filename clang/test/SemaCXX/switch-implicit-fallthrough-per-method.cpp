// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough-per-function %s


int fallthrough(int n) {
  switch (n / 10) {
    case 0:
      n += 100;
    case 1:  // expected-warning{{unannotated fall-through}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      switch (n) {
      case 111:
        n += 111;
        [[clang::fallthrough]];
      case 112:
        n += 112;
      case 113:  // expected-warning{{unannotated fall-through}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
        n += 113;
        break    ;
      }
  }
  return n;
}

int fallthrough2(int n) {
  switch (n / 10) {
    case 0:
      n += 100;
    case 1:  // no warning, as we didn't "opt-in" for it in this method
      switch (n) {
      case 111:
        n += 111;
      case 112:  // no warning, as we didn't "opt-in" for it in this method
        n += 112;
      case 113:  // no warning, as we didn't "opt-in" for it in this method
        n += 113;
        break    ;
      }
  }
  return n;
}

void unscoped(int n) {
  switch (n % 2) {
    case 0:
      [[fallthrough]];
    case 2:
      [[clang::fallthrough]];
    case 1:
      break;
  }
}
