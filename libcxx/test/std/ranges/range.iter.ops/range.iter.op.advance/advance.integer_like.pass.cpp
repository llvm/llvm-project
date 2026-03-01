#include <ranges>
#include <cassert>

int main() {
  int buffer[10];
  int* it = buffer;

  std::ranges::advance(it, 3);

  assert(it == buffer + 3);
}
