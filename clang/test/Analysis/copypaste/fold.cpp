// RUN: %clang_analyze_cc1 -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// expected-no-diagnostics

int global = 0;

template<typename ...Args>
int foo1(Args&&... args) {
  if (global > 0)
    return 0;
  else if (global < 0)
    return (args + ...);
  return 1;
}

// Different opeator in fold expression.
template<typename ...Args>
int foo2(Args&&... args) {
  if (global > 0)
    return 0;
  else if (global < 0)
    return (args - ...);
  return 1;
}

// Parameter pack on a different side
template<typename ...Args>
int foo3(Args&&... args) {
  if (global > 0)
    return 0;
  else if (global < 0)
    return -1;
  return (... + args);
return 1;
}
