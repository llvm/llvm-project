// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -verify %s

// expected-no-diagnostics

struct toggle {
  bool value;
};

toggle global_toggle;
toggle get_global_toggle() { return global_toggle; }

int oob_access();

bool compare(toggle one, bool other) {
  if (one.value != other)
    return true;

  if (one.value)
    oob_access();
  return true;
}

bool coin();

void bar() {
  bool left = coin();
  bool right = coin();
  for (;;)
    compare(get_global_toggle(), left) && compare(get_global_toggle(), right);
}
