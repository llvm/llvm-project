// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=alpha.clone.CloneChecker \
// RUN:   -analyzer-config alpha.clone.CloneChecker:ReportNormalClones=false \
// RUN:   -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10

// Tests finding a suspicious clone that references local variables.

void log();

int max(int a, int b) {
  log();
  if (a > b)
    return a;
  return b; // expected-note{{Similar code using 'b' here}}
}

int maxClone(int x, int y, int z) {
  log();
  if (x > y)
    return x;
  return z; // expected-warning{{Potential copy-paste error; did you really mean to use 'z' here?}}
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
      m1.unlock(); // expected-note{{Similar code using 'm1' here}}
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
      m2.unlock();  // expected-warning{{Potential copy-paste error; did you really mean to use 'm2' here?}}
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
  c -= b * a; // expected-warning{{Potential copy-paste error; did you really mean to use 'b' here?}}
  return c;
}

int fooClone(int a, int b, int c) {
  a += b + c;
  b /= a + b;
  c -= a * a; // expected-note{{Similar code using 'a' here}}
  return c;
}


// Tests that for clone groups with a many possible suspicious clone pairs, at
// most one warning per clone group is generated and every relevant clone is
// reported through either a warning or a note.

long bar1(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = b * b - c; // expected-warning{{Potential copy-paste error; did you really mean to use 'b' here?}}
  return d;
}

long bar2(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = c * b - c; // expected-note{{Similar code using 'c' here}} \
                 // expected-warning{{Potential copy-paste error; did you really mean to use 'c' here?}}
  return d;
}

long bar3(long a, long b, long c, long d) {
  c = a - b;
  c = c / d * a;
  d = a * b - c; // expected-note{{Similar code using 'a' here}}
  return d;
}
