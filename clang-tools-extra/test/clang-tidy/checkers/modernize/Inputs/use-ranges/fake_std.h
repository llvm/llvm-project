#ifndef USE_RANGES_FAKE_STD_H
#define USE_RANGES_FAKE_STD_H

#include <vector>

namespace std {

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

// Copy
template <class InputIt, class OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first);
template <class InputIt, class OutputIt, class UnaryPred>
OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first,
                 UnaryPred pred) {
  return d_first;
}
template <class BidirIt1, class BidirIt2>
BidirIt2 copy_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);

// Move
template <class InputIt, class OutputIt>
OutputIt move(InputIt first, InputIt last, OutputIt d_first);
template <class BidirIt1, class BidirIt2>
BidirIt2 move_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);

// Reverse
template <typename Iter> void reverse(Iter begin, Iter end);
template <class BidirIt, class OutputIt>
OutputIt reverse_copy(BidirIt first, BidirIt last, OutputIt d_first);

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
ForwardIt unique(ForwardIt first, ForwardIt last);
template <class ForwardIt, class BinaryPred>
ForwardIt unique(ForwardIt first, ForwardIt last, BinaryPred pred) {
  return first;
}

template <class ForwardIt, class T>
ForwardIt remove(ForwardIt first, ForwardIt last, const T &value);
template <class ForwardIt, class UnaryPred>
ForwardIt remove_if(ForwardIt first, ForwardIt last, UnaryPred pred) {
  return first;
}
template <class InputIt, class OutputIt, class T>
OutputIt remove_copy(InputIt first, InputIt last, OutputIt d_first,
                     const T &value);
template <class InputIt, class OutputIt, class UnaryPred>
OutputIt remove_copy_if(InputIt first, InputIt last, OutputIt d_first,
                        UnaryPred pred) {
  return d_first;
}

template <class ForwardIt, class UnaryPred>
ForwardIt partition(ForwardIt first, ForwardIt last, UnaryPred pred) {
  return first;
}
template <class BidirIt, class UnaryPred>
BidirIt stable_partition(BidirIt first, BidirIt last, UnaryPred pred) {
  return first;
}

template <class ForwardIt>
ForwardIt rotate(ForwardIt first, ForwardIt middle, ForwardIt last);
template <class InputIt, class OutputIt>
OutputIt unique_copy(InputIt first, InputIt last, OutputIt d_first);
template <class InputIt, class OutputIt, class UnaryOp>
OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
  return d_first;
}
template <class InputIt1, class InputIt2, class OutputIt>
OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2,
               InputIt2 last2, OutputIt d_first);
template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   InputIt2 last2, OutputIt d_first);
template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                          InputIt2 last2, OutputIt d_first);
template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_difference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        InputIt2 last2, OutputIt d_first);
template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_symmetric_difference(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2,
                                  OutputIt d_first);
template <class InputIt, class RandomIt>
RandomIt partial_sort_copy(InputIt first, InputIt last, RandomIt d_first,
                           RandomIt d_last);
template <class InputIt, class ForwardIt>
ForwardIt uninitialized_copy(InputIt first, InputIt last, ForwardIt d_first);
template <class InputIt, class ForwardIt>
ForwardIt uninitialized_move(InputIt first, InputIt last, ForwardIt d_first);
template <class ForwardIt, class OutputIt>
OutputIt rotate_copy(ForwardIt first, ForwardIt middle, ForwardIt last,
                     OutputIt d_first);
} // namespace _V1

} // namespace std

#endif // USE_RANGES_FAKE_STD_H
