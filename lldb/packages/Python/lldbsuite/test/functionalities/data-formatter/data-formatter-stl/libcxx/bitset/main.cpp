#include <bitset>

template<std::size_t N>
void fill(std::bitset<N> &b) {
  b.set();
  b[0] = b[1] = false;
  for (std::size_t i = 2; i < N; ++i) {
    for (std::size_t j = 2*i; j < N; j+=i)
      b[j] = false;
  }
}

int main() {
  std::bitset<0> empty;
  std::bitset<13> small;
  fill(small);
  std::bitset<200> large;
  fill(large);
  return 0; // break here
}
