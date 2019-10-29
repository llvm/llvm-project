#include <cstdlib>
#include <utility>

int main(int argc, char **argv) {
  std::size_t f = argc;
  f = std::abs(argc);
  f = std::div(argc * 2, argc).quot;
  std::swap(f, f);
  return f; // Set break point at this line.
}
