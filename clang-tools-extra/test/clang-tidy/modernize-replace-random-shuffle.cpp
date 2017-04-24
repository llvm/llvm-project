// RUN: %check_clang_tidy %s modernize-replace-random-shuffle %t -- -- -std=c++11

//CHECK-FIXES: #include <random>

namespace std {
template <typename T> struct vec_iterator {
  T *ptr;
  vec_iterator operator++(int);
};

template <typename T> struct vector {
  typedef vec_iterator<T> iterator;

  iterator begin();
  iterator end();
};

template <typename FwIt>
void random_shuffle(FwIt begin, FwIt end);

template <typename FwIt, typename randomFunc>
void random_shuffle(FwIt begin, FwIt end, randomFunc& randomfunc);

template <typename FwIt>
void shuffle(FwIt begin, FwIt end);
} // namespace std

// Random Func
int myrandom (int i) { return i;}

using namespace std;

int main() {
  std::vector<int> vec;

  std::random_shuffle(vec.begin(), vec.end());
  // CHECK-MESSAGE: [[@LINE-1]]:3: warning: 'std::random_shuffle' has been removed in C++17; use 'std::shuffle' instead
  // CHECK-FIXES: std::shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));

  std::shuffle(vec.begin(), vec.end());

  random_shuffle(vec.begin(), vec.end());
  // CHECK-MESSAGE: [[@LINE-1]]:3: warning: 'std::random_shuffle' has been removed in C++17; use 'std::shuffle' instead
  // CHECK-FIXES: shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));
  
  std::random_shuffle(vec.begin(), vec.end(), myrandom);
  // CHECK-MESSAGE: [[@LINE-1]]:3: warning: 'std::random_shuffle' has been removed in C++17; use 'std::shuffle' and an alternative random mechanism instead
  // CHECK-FIXES: std::shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));

  random_shuffle(vec.begin(), vec.end(), myrandom);
  // CHECK-MESSAGE: [[@LINE-1]]:3: warning: 'std::random_shuffle' has been removed in C++17; use 'std::shuffle' and an alternative random mechanism instead
  // CHECK-FIXES: shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));

  shuffle(vec.begin(), vec.end());

  return 0;
}
