// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:ReportSuspiciousClones=true  -analyzer-config alpha.clone.CloneChecker:ReportNormalClones=false -verify %s

// Tests finding a suspicious clone that references local variables.

void log();

int max(int a, int b) {
  log();
  if (a > b)
    return a;
  return b; // expected-note{{suggestion is based on the usage of this variable in a similar piece of code}}
}

int maxClone(int x, int y, int z) {
  log();
  if (x > y)
    return x;
  return z; // expected-warning{{suspicious code clone detected; did you mean to use 'y'?}}
}

// Tests finding a suspicious clone that references global variables.

struct mutex {
  bool try_lock();
  void unlock();
};

mutex m1;
mutex m2;
int i;

void busyIncrement() {
  while (true) {
    if (m1.try_lock()) {
      ++i;
      m1.unlock(); // expected-note{{suggestion is based on the usage of this variable in a similar piece of code}}
      if (i > 1000) {
        return;
      }
    }
  }
}

void faultyBusyIncrement() {
  while (true) {
    if (m1.try_lock()) {
      ++i;
      m2.unlock();  // expected-warning{{suspicious code clone detected; did you mean to use 'm1'?}}
      if (i > 1000) {
        return;
      }
    }
  }
}

// Tests that we provide two suggestions in cases where two fixes are possible.

int foo(int a, int b, int c) {
  a += b + c;
  b /= a + b;
  c -= b * a; // expected-warning{{suspicious code clone detected; did you mean to use 'a'?}}
  return c;
}

int fooClone(int a, int b, int c) {
  a += b + c;
  b /= a + b;
  c -= a * a; // expected-note{{suggestion is based on the usage of this variable in a similar piece of code; did you mean to use 'b'?}}
  return c;
}


// Tests that for clone groups with a many possible suspicious clone pairs, at
// most one warning per clone group is generated and every relevant clone is
// reported through either a warning or a note.

long bar1(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = b * b - c; // expected-warning{{suspicious code clone detected; did you mean to use 'c'?}}
  return d;
}

long bar2(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = c * b - c; // expected-note{{suggestion is based on the usage of this variable in a similar piece of code; did you mean to use 'b'?}} \
                 // expected-warning{{suspicious code clone detected; did you mean to use 'a'?}}
  return d;
}

long bar3(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = a * b - c; // expected-note{{suggestion is based on the usage of this variable in a similar piece of code; did you mean to use 'c'?}}
  return d;
}
