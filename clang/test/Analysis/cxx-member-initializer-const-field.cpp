// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// This tests false-positive issues related to PR48534.
//
// Essentially, having a default member initializer for a constant member does
// not necessarily imply the member will have the given default value.

struct WithConstructor {
  int *const ptr = nullptr;
  WithConstructor(int *x) : ptr(x) {}

  static auto compliant() {
    WithConstructor c(new int{});
    return *(c.ptr); // no warning
  }

  static auto compliantWithParam(WithConstructor c) {
    return *(c.ptr); // no warning
  }

  static auto issue() {
    WithConstructor c(nullptr);
    return *(c.ptr); // expected-warning{{Dereference of null pointer (loaded from field 'ptr')}}
  }
};

struct RegularAggregate {
  int *const ptr = nullptr;

  static int compliant() {
    RegularAggregate c{new int{}};
    return *(c.ptr); // no warning
  }

  static int issue() {
    RegularAggregate c;
    return *(c.ptr); // expected-warning{{Dereference of null pointer (loaded from field 'ptr')}}
  }
};

struct WithConstructorAndArithmetic {
  int const i = 0;
  WithConstructorAndArithmetic(int x) : i(x + 1) {}

  static int compliant(int y) {
    WithConstructorAndArithmetic c(0);
    return y / c.i; // no warning
  }

  static int issue(int y) {
    WithConstructorAndArithmetic c(-1);
    return y / c.i; // expected-warning{{Division by zero}}
  }
};

struct WithConstructorDeclarationOnly {
  int const i = 0;
  WithConstructorDeclarationOnly(int x); // definition not visible.

  static int compliant1(int y) {
    WithConstructorDeclarationOnly c(0);
    return y / c.i; // no warning
  }

  static int compliant2(int y) {
    WithConstructorDeclarationOnly c(-1);
    return y / c.i; // no warning
  }
};

// NonAggregateFP is not an aggregate (j is a private non-static field) and has no custom constructor.
// So we know i and j will always be 0 and 42, respectively.
// That being said, this is not implemented because it is deemed too rare to be worth the complexity.
struct NonAggregateFP {
public:
  int const i = 0;

private:
  int const j = 42;

public:
  static int falsePositive1(NonAggregateFP c) {
    return 10 / c.i; // FIXME: Currently, no warning.
  }

  static int falsePositive2(NonAggregateFP c) {
    return 10 / (c.j - 42); // FIXME: Currently, no warning.
  }
};

struct NonAggregate {
public:
  int const i = 0;

private:
  int const j = 42;

  NonAggregate(NonAggregate const &); // not provided, could set i and j to arbitrary values.

public:
  static int compliant1(NonAggregate c) {
    return 10 / c.i; // no warning
  }

  static int compliant2(NonAggregate c) {
    return 10 / (c.j - 42); // no warning
  }
};

struct WithStaticMember {
  static int const i = 0;

  static int issue1(WithStaticMember c) {
    return 10 / c.i; // expected-warning{{division by zero is undefined}} expected-warning{{Division by zero}}
  }

  static int issue2() {
    return 10 / WithStaticMember::i; // expected-warning{{division by zero is undefined}} expected-warning{{Division by zero}}
  }
};
