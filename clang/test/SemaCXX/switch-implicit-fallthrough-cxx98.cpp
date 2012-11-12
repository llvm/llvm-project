// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -Wimplicit-fallthrough %s


int fallthrough(int n) {
  switch (n / 10) {
      if (n - 1) {
        n = 100;
      } else if (n - 2) {
        n = 101;
      } else if (n - 3) {
        n = 102;
      }
    case -1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
    case 0: {// expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
    }
    case 1:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      n += 100         ;
    case 3:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      if (n > 0)
        n += 200;
    case 4:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      if (n < 0)
        ;
    case 5:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      switch (n) {
      case 111:
        break;
      case 112:
        break;
      case 113:
        break    ;
      }
    case 6:  // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'break;' to avoid fall-through}}
      n += 300;
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
