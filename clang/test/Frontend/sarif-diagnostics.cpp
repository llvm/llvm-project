// RUN: %clang -fsyntax-only -Wall -Wextra -fdiagnostics-format=sarif %s > %t 2>&1 || true
// RUN: cat %t | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-diagnostics.cpp.sarif -

// FIXME: this test is incredibly fragile because the `main()` function
// must be on line 12 in order for the line numbers in the SARIF output
// to match the expected values
//
// So these comment lines are being used to ensure the code below happens
// to work properly for the test coverage, which as you can imagine, is not
// the best way to structure the test. We should consider having a way to
// tag line numbers in the test source to match in the SARIF output.
void main() {
  int i = hello;

  float test = 1a.0;

  if (true)
    bool Yes = true;
    return;

  bool j = hi;
}
}

struct t1 { };
void f1(t1 x, t1 y) {
    x + y;
}
