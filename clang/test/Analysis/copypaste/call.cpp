// RUN: %clang_cc1 -analyze -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

bool a();
bool b();

// Calls method a with some extra code to pass the minimum complexity
bool foo1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return a();
  return true;
}

// Calls method b with some extra code to pass the minimum complexity
bool foo2(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return b();
  return true;
}

// Test that we don't crash on function pointer calls

bool (*funcPtr)(int);

bool fooPtr1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return funcPtr(1);
  return true;
}
