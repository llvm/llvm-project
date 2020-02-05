// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify -Wimplicit-fallthrough %s

int fallthrough_comment(int n) {
  switch (n) {
  case 0:
    n++;
    // FALLTHROUGH
  case 1:
    n++;

    /*fall-through.*/

  case 2:
    n++;
  case 3: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '__attribute__((fallthrough));' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
    n++;
    break;
  }
  return n;
}
