// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t
// RUN: %check_clang_tidy -std=c++23 %s modernize-use-ranges %t -check-suffixes=,CPP23

// CHECK-FIXES: #include <algorithm>
// CHECK-FIXES-CPP23: #include <numeric>
// CHECK-FIXES: #include <ranges>

namespace std {

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = T*;
  using reverse_const_iterator = const T*;

  constexpr const_iterator begin() const;
  constexpr const_iterator end() const;
  constexpr const_iterator cbegin() const;
  constexpr const_iterator cend() const;
  constexpr iterator begin();
  constexpr iterator end();
  constexpr reverse_const_iterator rbegin() const;
  constexpr reverse_const_iterator rend() const;
  constexpr reverse_const_iterator crbegin() const;
  constexpr reverse_const_iterator crend() const;
  constexpr reverse_iterator rbegin();
  constexpr reverse_iterator rend();
};

template <typename Container> constexpr auto begin(const Container &Cont) {
  return Cont.begin();
}

template <typename Container> constexpr auto begin(Container &Cont) {
  return Cont.begin();
}

template <typename Container> constexpr auto end(const Container &Cont) {
  return Cont.end();
}

template <typename Container> constexpr auto end(Container &Cont) {
  return Cont.end();
}

template <typename Container> constexpr auto cbegin(const Container &Cont) {
  return Cont.cbegin();
}

template <typename Container> constexpr auto cend(const Container &Cont) {
  return Cont.cend();
}

template <typename Container> constexpr auto rbegin(const Container &Cont) {
  return Cont.rbegin();
}

template <typename Container> constexpr auto rbegin(Container &Cont) {
  return Cont.rbegin();
}

template <typename Container> constexpr auto rend(const Container &Cont) {
  return Cont.rend();
}

template <typename Container> constexpr auto rend(Container &Cont) {
  return Cont.rend();
}

template <typename Container> constexpr auto crbegin(const Container &Cont) {
  return Cont.crbegin();
}

template <typename Container> constexpr auto crend(const Container &Cont) {
  return Cont.crend();
}
// Find
template< class InputIt, class T >
InputIt find( InputIt first, InputIt last, const T& value );

// Reverse
template <typename Iter> void reverse(Iter begin, Iter end);

// Includes
template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

// IsPermutation
template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);
template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

// Equal
template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class InputIt1, class InputIt2, class BinaryPred>
bool equal(InputIt1 first1, InputIt1 last1,
           InputIt2 first2, InputIt2 last2, BinaryPred p) {
  // Need a definition to suppress undefined_internal_type when invoked with lambda
  return true;
}

template <class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value);

} // namespace std

void Positives() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 0);

  std::find(I.cbegin(), I.cend(), 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 1);

  std::find(std::begin(I), std::end(I), 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 2);

  std::find(std::cbegin(I), std::cend(I), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 3);

  std::find(std::cbegin(I), I.cend(), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 4);

  std::reverse(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::reverse(I);

  std::includes(I.begin(), I.end(), I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, I);

  std::includes(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, J);

  std::is_permutation(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::is_permutation(I, J);

  std::equal(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J);

  std::equal(I.begin(), I.end(), J.begin(), J.end(), [](int a, int b){ return a == b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J, [](int a, int b){ return a == b; });

  std::iota(I.begin(), I.end(), 0);
  // CHECK-MESSAGES-CPP23: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES-CPP23: std::ranges::iota(I, 0);

  using std::find;
  namespace my_std = std;

  // Potentially these could be updated to better qualify the replaced function name
  find(I.begin(), I.end(), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 5);

  my_std::find(I.begin(), I.end(), 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 6);
}

void Reverse(){
  std::vector<int> I, J;
  std::find(I.rbegin(), I.rend(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(std::views::reverse(I), 0);

  std::equal(std::rbegin(I), std::rend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(std::views::reverse(I), J);

  std::equal(I.begin(), I.end(), std::crbegin(J), std::crend(J));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, std::views::reverse(J));
}

void Negatives() {
  std::vector<int> I, J;
  std::find(I.begin(), J.end(), 0);
  std::find(I.begin(), I.begin(), 0);
  std::find(I.end(), I.begin(), 0);


  // Need both ranges for this one
  std::is_permutation(I.begin(), I.end(), J.begin());

  // We only have one valid match here and the ranges::equal function needs 2 complete ranges
  std::equal(I.begin(), I.end(), J.begin());
  std::equal(I.begin(), I.end(), J.end(), J.end());
  std::equal(std::rbegin(I), std::rend(I), std::rend(J), std::rbegin(J));
  std::equal(I.begin(), J.end(), I.begin(), I.end());
}
