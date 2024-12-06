#ifndef USE_RANGES_FAKE_STD_H
#define USE_RANGES_FAKE_STD_H

namespace std {

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = T *;
  using reverse_const_iterator = const T *;

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
template <class InputIt, class T>
InputIt find(InputIt first, InputIt last, const T &value);

// Reverse
template <typename Iter> void reverse(Iter begin, Iter end);

// Includes
template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

inline namespace _V1 {
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
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2,
           BinaryPred p) {
  // Need a definition to suppress undefined_internal_type when invoked with
  // lambda
  return true;
}

template <class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value);

template <class ForwardIt>
ForwardIt rotate(ForwardIt first, ForwardIt middle, ForwardIt last);
} // namespace _V1

} // namespace std

#endif // USE_RANGES_FAKE_STD_H
