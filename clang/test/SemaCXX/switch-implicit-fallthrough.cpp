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
    case -1:  // no warning here, ignore fall-through from unreachable code
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
    case 67:
    case 68:
      break;
  }
  switch (n / 15) {
label_default:
    default:
      n += 333;
      if (n % 10)
        goto label_default;
      break;
    case 70:
      n += 335;
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

void fallthrough3(int n) {
  switch (n) {
    case 1:
      do {
        return;
      } while (0);
    case 2:
      do {
        ClassWithDtor temp;
        return;
      } while (0);
    case 3:
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

void fallthrough_cfgblock_with_null_successor(int x) {
  (x && "") ? (void)(0) : (void)(1);
  switch (x) {}
}

int fallthrough_position(int n) {
  switch (n) {
      n += 300;
      [[clang::fallthrough]];  // expected-warning{{fallthrough annotation in unreachable code}}
    case 221:
      return 1;
      [[clang::fallthrough]];  // expected-warning{{fallthrough annotation in unreachable code}}
    case 222:
      n += 400;
    case 223:          // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
  }

  long p = static_cast<long>(n) * n;
  switch (sizeof(p)) {
    case 9:
      n += static_cast<int>(p >> 32);
      [[clang::fallthrough]];  // no warning here
    case 5:
      n += static_cast<int>(p);
      [[clang::fallthrough]];  // no warning here
    default:
      n += 1;
      break;
  }

  return n;
}

enum Enum {
  Value1, Value2
};

int fallthrough_covered_enums(Enum e) {
  int n = 0;
  switch (e) {
    default:
      n += 17;
      [[clang::fallthrough]];  // no warning here, this shouldn't be treated as unreachable code
    case Value1:
      n += 19;
      break;
    case Value2:
      n += 21;
      break;
  }
  return n;
}

// Fallthrough annotations in local classes used to generate "fallthrough
// annotation does not directly precede switch label" warning.
void fallthrough_in_local_class() {
  class C {
    void f(int x) {
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
    }
  };
}

// Fallthrough annotations in lambdas used to generate "fallthrough
// annotation does not directly precede switch label" warning.
void fallthrough_in_lambda() {
  (void)[] {
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
}

namespace PR18983 {
  void fatal() __attribute__((noreturn));
  int num();
  void test() {
    switch (num()) {
    case 1:
      fatal();
      // Don't issue a warning.
    case 2:
      break;
    }
  }
}

int fallthrough_placement_error(int n) {
  switch (n) {
      [[clang::fallthrough]]; // expected-warning{{fallthrough annotation in unreachable code}}
      n += 300;
    case 221:
      [[clang::fallthrough]]; // expected-error{{fallthrough annotation does not directly precede switch label}}
      return 1;
    case 222:
      [[clang::fallthrough]]; // expected-error{{fallthrough annotation does not directly precede switch label}}
      n += 400;
      [[clang::fallthrough]];
    case 223:
      [[clang::fallthrough]]; // expected-error{{fallthrough annotation does not directly precede switch label}}
  }
  return n;
}

int fallthrough_targets(int n) {
  [[clang::fallthrough]]; // expected-error{{fallthrough annotation is outside switch statement}}

  [[clang::fallthrough]]  // expected-error{{'fallthrough' attribute only applies to empty statements}}
  switch (n) {
    case 121:
      n += 400;
      [[clang::fallthrough]]; // no warning here, correct target
    case 123:
      [[clang::fallthrough]]  // expected-error{{'fallthrough' attribute only applies to empty statements}}
      n += 800;
      break;
    [[clang::fallthrough]]    // expected-error{{'fallthrough' attribute is only allowed on empty statements}} expected-note{{did you forget ';'?}}
    case 125:
      n += 1600;
  }
  return n;
}

int fallthrough_alt_spelling(int n) {
  switch (n) {
  case 0:
    n++;
    [[clang::fallthrough]];
  case 1:
    n++;
    [[clang::__fallthrough__]];
  case 2:
    n++;
    break;
  }
  return n;
}

int fallthrough_attribute_spelling(int n) {
  switch (n) {
  case 0:
    n++;
    __attribute__((fallthrough));
  case 1:
    n++;
    break;
  }
  return n;
}
