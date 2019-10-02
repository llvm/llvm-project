#include <utility>

int main(int argc, char **argv) {
  std::pair<int, long> pair = std::make_pair<int, float>(1, 2L);
  return pair.first; // Set break point at this line.
}
