// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// expected-no-diagnostics

#include "Inputs/no-store-suppression.h"

using namespace std;

namespace value_uninitialized_after_stream_shift {
void use(char c);

// Technically, it is absolutely necessary to check the status of cin after
// read before using the value that just read from it. Practically, we don't
// really care unless we eventually come up with a special security check
// for just that purpose. Static Analyzer shouldn't be yelling at every person's
// third program in their C++ 101.
void foo() {
  char c;
  std::cin >> c;
  use(c); // no-warning
}
} // namespace value_uninitialized_after_stream_shift
