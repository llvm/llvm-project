// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough %s


int fallthrough(int n) {
  switch (n / 10) {
      if (n - 1) {
        n = 100;
      } else if (n - 2) {
        n = 101;
      } else if (n - 3) {
        n = 102;
      }
    case -1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
    case 0: {// expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
    }
    case 1:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      n += 100         ;
    case 3:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      if (n > 0)
        n += 200;
    case 4:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      if (n < 0)
        ;
    case 5:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      switch (n) {
      case 111:
        break;
      case 112:
        break;
      case 113:
        break    ;
      }
    case 6:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      n += 300;
    case 66:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      break;
  }
  switch (n / 20) {
    case 7:
      n += 400;
      [[clang::fallthrough]];
    case 9:  // no warning here, intended fall-through marked with an attribute
      n += 800;
      [[clang::fallthrough]];
    default: { // no warning here, intended fall-through marked with an attribute
      if (n % 2 == 0) {
        return 1;
      } else {
        [[clang::fallthrough]];
      }
    }
    case 10:  // no warning here, intended fall-through marked with an attribute
      if (n % 3 == 0) {
        n %= 3;
      } else {
        [[clang::fallthrough]];
      }
    case 110:  // expected-warning{{unannotated fall-through between switch labels}} but no fix-it hint as we have one fall-through annotation!
      n += 800;
  }
  switch (n / 30) {
    case 11:
    case 12:  // no warning here, intended fall-through, no statement between labels
      n += 1600;
  }
  switch (n / 40) {
    case 13:
      if (n % 2 == 0) {
        return 1;
      } else {
        return 2;
      }
    case 15:  // no warning here, there's no fall-through
      n += 3200;
  }
  switch (n / 50) {
    case 17: {
      if (n % 2 == 0) {
        return 1;
      } else {
        return 2;
      }
    }
    case 19: { // no warning here, there's no fall-through
      n += 6400;
      return 3;
    }
    case 21: { // no warning here, there's no fall-through
      break;
    }
    case 23: // no warning here, there's no fall-through
      n += 128000;
      break;
    case 25: // no warning here, there's no fall-through
      break;
  }

  return n;
}

class ClassWithDtor {
public:
  ~ClassWithDtor() {}
};

void fallthrough2(int n) {
  switch (n) {
    case 0:
    {
      ClassWithDtor temp;
      break;
    }
    default: // no warning here, there's no fall-through
      break;
  }
}

#define MY_SWITCH(X, Y, Z, U, V) switch (X) { case Y: Z; case U: V; }
#define MY_SWITCH2(X, Y, Z) switch (X) { Y; Z; }
#define MY_CASE(X, Y) case X: Y
#define MY_CASE2(X, Y, U, V) case X: Y; case U: V

int fallthrough_macro1(int n) {
  MY_SWITCH(n, 13, n *= 2, 14, break)  // expected-warning{{unannotated fall-through between switch labels}}

  switch (n + 1) {
    MY_CASE(33, n += 2);
    MY_CASE(44, break);  // expected-warning{{unannotated fall-through between switch labels}}
    MY_CASE(55, n += 3);
  }

  switch (n + 3) {
    MY_CASE(333, return 333);
    MY_CASE2(444, n += 44, 4444, break);  // expected-warning{{unannotated fall-through between switch labels}}
    MY_CASE(555, n += 33);
  }

  MY_SWITCH2(n + 4, MY_CASE(17, n *= 3), MY_CASE(19, break))  // expected-warning{{unannotated fall-through between switch labels}}

  MY_SWITCH2(n + 5, MY_CASE(21, break), MY_CASE2(23, n *= 7, 25, break))  // expected-warning{{unannotated fall-through between switch labels}}

  return n;
}

int fallthrough_position(int n) {
  switch (n) {
      [[clang::fallthrough]];  // expected-warning{{fallthrough annotation in unreachable code}}
    case 221:
      [[clang::fallthrough]]; // expected-warning{{fallthrough annotation does not directly precede switch label}}
      return 1;
      [[clang::fallthrough]];  // expected-warning{{fallthrough annotation in unreachable code}}
    case 222:
      [[clang::fallthrough]]; // expected-warning{{fallthrough annotation does not directly precede switch label}}
      n += 400;
    case 223:          // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      [[clang::fallthrough]]; // expected-warning{{fallthrough annotation does not directly precede switch label}}
  }

  // TODO: uncomment this test after CFG gets more options to deal with
  // unreachable code:
  // http://lists.cs.uiuc.edu/pipermail/cfe-commits/Week-of-Mon-20120507/057370.html
#if 0
  long p = static_cast<long>(n) * n;
  switch (sizeof(p)) {
    case 9:                    // this test will not work on compilers with 72-bit long
      n += static_cast<int>(p >> 32);
      [[clang::fallthrough]];  // no warning here
    case 5:                    // it is not intended to work on compilers with 40-bit long as well
      n += static_cast<int>(p);
      break;
    default:
     break;
  }
#endif

  return n;
}

int fallthrough_targets(int n) {
  [[clang::fallthrough]]; // expected-error{{fallthrough annotation is outside switch statement}}

  [[clang::fallthrough]]  // expected-error{{fallthrough attribute is only allowed on empty statements}}
  switch (n) {
    case 121:
      n += 400;
      [[clang::fallthrough]]; // no warning here, correct target
    case 123:
      [[clang::fallthrough]]  // expected-error{{fallthrough attribute is only allowed on empty statements}}
      n += 800;
      break;
    [[clang::fallthrough]]    // expected-error{{fallthrough attribute is only allowed on empty statements}} expected-note{{did you forget ';'?}}
    case 125:
      n += 1600;
  }
  return n;
}
