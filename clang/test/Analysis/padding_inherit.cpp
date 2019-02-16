// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=optin.performance -analyzer-config optin.performance.Padding:AllowedPad=20 -verify %s

// A class that has no fields and one base class should visit that base class
// instead. Note that despite having excess padding of 2, this is flagged
// because of its usage in an array of 100 elements below (`ais').
// TODO: Add a note to the bug report with BugReport::addNote() to mention the
// variable using the class and also mention what class is inherting from what.
// expected-warning@+1{{Excessive padding in 'struct FakeIntSandwich'}}
struct FakeIntSandwich {
  char c1;
  int i;
  char c2;
};

struct AnotherIntSandwich : FakeIntSandwich { // no-warning
};

// But we don't yet support multiple base classes.
struct IntSandwich {};
struct TooManyBaseClasses : FakeIntSandwich, IntSandwich { // no-warning
};

AnotherIntSandwich ais[100];

struct Empty {};
struct DoubleEmpty : Empty { // no-warning
    Empty e;
};
