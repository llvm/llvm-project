#ifndef USE_RANGES_FAKE_STD_H
#define USE_RANGES_FAKE_STD_H
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
// Find
template< class InputIt, class T >
InputIt find(InputIt first, InputIt last, const T& value);

template <typename Iter> void reverse(Iter begin, Iter end);

template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

template <class BidirIt>
bool next_permutation(BidirIt first, BidirIt last);

inline namespace inline_test{

template <class ForwardIt1, class ForwardIt2>
bool equal(ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2);

template <class RandomIt>
void push_heap(RandomIt first, RandomIt last);

template <class InputIt, class OutputIt, class UnaryPred>
OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPred pred);

template <class ForwardIt>
ForwardIt is_sorted_until(ForwardIt first, ForwardIt last);

template <class InputIt>
void reduce(InputIt first, InputIt last);

template <class InputIt, class T>
T reduce(InputIt first, InputIt last, T init);

template <class InputIt, class T, class BinaryOp>
T reduce(InputIt first, InputIt last, T init, BinaryOp op) {
  // Need a definition to suppress undefined_internal_type when invoked with lambda
  return init;
}

template <class InputIt, class T>
T accumulate(InputIt first, InputIt last, T init);

} // namespace inline_test

} // namespace std

#endif // USE_RANGES_FAKE_STD_H
