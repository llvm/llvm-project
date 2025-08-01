// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=cplusplus -verify %s
// expected-no-diagnostics

template <int ...args>
bool issue151529()
{
  [[assume (((args >= 0) && ...))]];
  return ((args >= 0) && ...);
}

int main() {
    issue151529();
    [[assume((true))]]; // crash
    return 0;
}
