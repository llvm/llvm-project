#include <ranges>
#include <cassert>

int main() {
  int buffer[10];
  int* it = buffer;

  auto result = std::ranges::next(it, 5);

  assert(result == buffer + 5);
}
