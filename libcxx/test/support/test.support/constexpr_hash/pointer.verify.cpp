

#include "constexpr_hash.h"

#include <cstddef>

constexpr std::size_t test_pointer() {
  typedef int* T;
  typedef support::constexpr_hash<T> H;

  //expected-error@+2 {{call to deleted constructor}}
  //expected-error@+1 {{ does not provide a call operator}}
  return (H()(T()));
}

int main(int, char**) {
  typedef int* T;
  typedef support::constexpr_hash<T> H;

  //expected-error@+2 {{call to deleted constructor}}
  //expected-error@+1 {{ does not provide a call operator}}
  std::size_t h = (H()(T()));

  //expected-error@+1 {{static assertion expression is not an integral constant expression}}
  static_assert(0 != test_pointer());

  return 0;
}