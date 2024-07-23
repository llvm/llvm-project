// RUN: %check_clang_tidy -std=c++14 %s bugprone-incorrect-iterators %t

namespace std {

template <class U, class V> struct pair {};

namespace execution {

class parallel_policy {};

constexpr parallel_policy par;

} // namespace execution

template <typename BiDirIter> class reverse_iterator {
public:
  constexpr explicit reverse_iterator(BiDirIter Iter);
  reverse_iterator operator+(int) const;
  reverse_iterator operator-(int) const;
};

template <typename InputIt> InputIt next(InputIt Iter, int n = 1);
template <typename BidirIt> BidirIt prev(BidirIt Iter, int n = 1);

template <typename BiDirIter>
reverse_iterator<BiDirIter> make_reverse_iterator(BiDirIter Iter);

template <typename T> class allocator {};

template <typename T, typename Allocator = allocator<T>> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using reverse_const_iterator = std::reverse_iterator<const_iterator>;

  vector() = default;
  vector(const vector &);
  vector(vector &&);
  template <typename InputIt>
  vector(InputIt first, InputIt last, const Allocator &alloc = Allocator());
  ~vector();

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

  template <class InputIt>
  iterator insert(const_iterator pos, InputIt first, InputIt last);
};

template <typename T> struct less {};

template <typename T> class __set_const_iterator {};

template <typename T> class __set_iterator : public __set_const_iterator<T> {};

template <typename Key, typename Compare = less<Key>,
          typename Allocator = allocator<Key>>
class set {
public:
  using iterator = __set_iterator<Key>;
  using const_iterator = __set_const_iterator<Key>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using reverse_const_iterator = std::reverse_iterator<const_iterator>;
  using value_type = Key;

  set();
  template <class InputIt>
  set(InputIt first, InputIt last, const Compare &comp = Compare(),
      const Allocator &alloc = Allocator());
  ~set();

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

  iterator insert(const_iterator pos, value_type &&value);
  template <class InputIt> void insert(InputIt first, InputIt last);
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

template <class Policy, class InputIt, class T>
InputIt find(Policy &&policy, InputIt first, InputIt last, const T &value);

template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   InputIt2 last2, OutputIt d_first);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class ForwardIt3>
ForwardIt3 set_union(ExecutionPolicy &&policy, ForwardIt1 first1,
                     ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2,
                     ForwardIt3 d_first);

template <class RandomIt> void push_heap(RandomIt first, RandomIt last);

template <class BidirIt1, class BidirIt2>
BidirIt2 copy_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);

template <class ForwardIt1, class ForwardIt2>
ForwardIt1 find_end(ForwardIt1 first, ForwardIt1 last, ForwardIt2 s_first,
                    ForwardIt2 s_last);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
ForwardIt1 find_end(ExecutionPolicy &&policy, ForwardIt1 first, ForwardIt1 last,
                    ForwardIt2 s_first, ForwardIt2 s_last);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
bool equal(ExecutionPolicy &&policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
bool equal(ExecutionPolicy &&policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

template <class InputIt, class NoThrowForwardIt>
NoThrowForwardIt uninitialized_copy(InputIt first, InputIt last,
                                    NoThrowForwardIt d_first);

template <class ExecutionPolicy, class ForwardIt, class NoThrowForwardIt>
NoThrowForwardIt uninitialized_copy(ExecutionPolicy &&policy, ForwardIt first,
                                    ForwardIt last, NoThrowForwardIt d_first);

template <class InputIt1, class InputIt2, class T>
T inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init);

template <class InputIt, class OutputIt>
OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first);

template <class InputIt, class OutputIt1, class OutputIt2, class UnaryPred>
pair<OutputIt1, OutputIt2> partition_copy(InputIt first, InputIt last,
                                          OutputIt1 d_first_true,
                                          OutputIt2 d_first_false, UnaryPred p);
template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class ForwardIt3, class UnaryPred>
pair<ForwardIt2, ForwardIt3>
partition_copy(ExecutionPolicy &&policy, ForwardIt1 first, ForwardIt1 last,
               ForwardIt2 d_first_true, ForwardIt3 d_first_false, UnaryPred p);

template <class InputIt, class OutputIt, class UnaryOp>
OutputIt transform(InputIt first1, InputIt last1, OutputIt d_first,
                   UnaryOp unary_op);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class UnaryOp>
ForwardIt2 transform(ExecutionPolicy &&policy, ForwardIt1 first1,
                     ForwardIt1 last1, ForwardIt2 d_first, UnaryOp unary_op);

template <class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   OutputIt d_first, BinaryOp binary_op);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class ForwardIt3, class BinaryOp>
ForwardIt3 transform(ExecutionPolicy &&policy, ForwardIt1 first1,
                     ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first,
                     BinaryOp binary_op);

template <class ForwardIt, class OutputIt>
OutputIt rotate_copy(ForwardIt first, ForwardIt n_first, ForwardIt last,
                     OutputIt d_first);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
ForwardIt2 rotate_copy(ExecutionPolicy &&policy, ForwardIt1 first,
                       ForwardIt1 n_first, ForwardIt1 last, ForwardIt2 d_first);

} // namespace std

template <typename T = int> static bool dummyUnary(const T &);
template <typename T = int, typename U = T>
static bool dummyBinary(const T &, const U &);

void Test() {
  std::vector<int> I;
  std::vector<int> J{I.begin(), I.begin()};
  // CHECK-NOTES: :[[@LINE-1]]:33: warning: 'begin' iterator supplied where an 'end' iterator is expected
  std::vector<int> K{I.begin(), J.end()};
  // CHECK-NOTES: [[@LINE-1]]:20: warning: mismatched ranges supplied to 'std::vector<int>::vector'
  // CHECK-NOTES: :[[@LINE-2]]:22: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:33: note: different range passed as the end iterator
  delete (new std::vector<int>(I.end(), I.end()));
  // CHECK-NOTES: :[[@LINE-1]]:32: warning: 'end' iterator supplied where a 'begin' iterator is expected
  auto RandomIIter = std::find(I.end(), I.begin(), 0);
  // CHECK-NOTES: :[[@LINE-1]]:32: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:41: warning: 'begin' iterator supplied where an 'end' iterator is expected
  std::find(std::execution::par, I.begin(), J.end(), 0);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::find'
  // CHECK-NOTES: :[[@LINE-2]]:34: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:45: note: different range passed as the end iterator
  std::find(std::make_reverse_iterator(I.end()),
            std::make_reverse_iterator(I.end()), 0);
  // CHECK-NOTES: :[[@LINE-1]]:40: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:13: note: 'make_reverse_iterator<int *>' changes 'end' into a 'begin' iterator
  std::find(I.rbegin(), std::make_reverse_iterator(J.begin()), 0);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::find'
  // CHECK-NOTES: :[[@LINE-2]]:13: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:52: note: different range passed as the end iterator

  I.insert(J.begin(), J.end(), J.begin());
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: 'insert<int *>' called with an iterator for a different container
  // CHECK-NOTES: :[[@LINE-2]]:3: note: container is specified here
  // CHECK-NOTES: :[[@LINE-3]]:12: note: different container provided here
  // CHECK-NOTES: :[[@LINE-4]]:23: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-5]]:32: warning: 'begin' iterator supplied where an 'end' iterator is expected

  std::set_union(I.begin(), I.begin(), J.end(), J.end(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:40: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:58: warning: 'end' iterator supplied where an output iterator is expected
  std::set_union(std::execution::par, I.begin(), I.begin(), J.end(), J.end(),
                 K.end());
  // CHECK-NOTES: :[[@LINE-2]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:61: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:18: warning: 'end' iterator supplied where an output iterator is expected

  std::push_heap(std::begin(I), std::end(J));
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::push_heap'
  // CHECK-NOTES: :[[@LINE-2]]:29: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:42: note: different range passed as the end iterator

  std::copy_backward(I.begin(), J.end(), K.begin());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::copy_backward'
  // CHECK-NOTES: :[[@LINE-2]]:22: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:33: note: different range passed as the end iterator
  // CHECK-NOTES: :[[@LINE-4]]:42: warning: 'begin' iterator supplied where an 'end' iterator is expected

  std::find_end(I.rbegin(), I.rbegin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:41: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::find_end(std::execution::par, I.rbegin(), I.rbegin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:62: warning: 'end' iterator supplied where a 'begin' iterator is expected

  std::equal(I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:34: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::equal(std::execution::par, I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:55: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::equal(I.begin(), I.end(), J.rbegin(), K.rend());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::equal'
  // CHECK-NOTES: :[[@LINE-2]]:34: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:46: note: different range passed as the end iterator
  std::equal(std::execution::par, I.begin(), I.end(), J.begin(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::equal'
  // CHECK-NOTES: :[[@LINE-2]]:55: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:66: note: different range passed as the end iterator

  std::is_permutation(I.begin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::is_permutation'
  // CHECK-NOTES: :[[@LINE-2]]:23: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:34: note: different range passed as the end iterator
  // CHECK-NOTES: :[[@LINE-4]]:43: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::is_permutation(I.begin(), I.end(), J.begin(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges supplied to 'std::is_permutation'
  // CHECK-NOTES: :[[@LINE-2]]:43: note: range passed as the begin iterator
  // CHECK-NOTES: :[[@LINE-3]]:54: note: different range passed as the end iterator

  std::uninitialized_copy(I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:47: warning: 'end' iterator supplied where an output iterator is expected
  std::uninitialized_copy(std::execution::par, I.begin(), I.begin(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:59: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:70: warning: 'end' iterator supplied where an output iterator is expected

  std::inner_product(std::make_reverse_iterator(I.begin()),
                     std::make_reverse_iterator(I.end()), J.end(), 0);
  // CHECK-NOTES: :[[@LINE-2]]:49: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:22: note: 'make_reverse_iterator<int *>' changes 'begin' into an 'end' iterator
  // CHECK-NOTES: :[[@LINE-3]]:49: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-4]]:22: note: 'make_reverse_iterator<int *>' changes 'end' into a 'begin' iterator
  // CHECK-NOTES: :[[@LINE-5]]:59: warning: 'end' iterator supplied where a 'begin' iterator is expected

  std::partial_sum(I.begin(), I.begin(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:31: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:42: warning: 'end' iterator supplied where an output iterator is expected

  std::partition_copy(I.begin(), I.begin(), J.end(), K.end(), &dummyUnary<>);
  // CHECK-NOTES: :[[@LINE-1]]:34: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:45: warning: 'end' iterator supplied where an output iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:54: warning: 'end' iterator supplied where an output iterator is expected
  std::partition_copy(std::execution::par, I.begin(), I.end(), J.end(), K.end(),
                      &dummyUnary<>);
  // CHECK-NOTES: :[[@LINE-2]]:64: warning: 'end' iterator supplied where an output iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:73: warning: 'end' iterator supplied where an output iterator is expected

  std::transform(I.begin(), I.begin(), J.end(), &dummyUnary<>);
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:40: warning: 'end' iterator supplied where an output iterator is expected
  std::transform(std::execution::par, I.begin(), I.begin(), J.end(),
                 &dummyUnary<>);
  // CHECK-NOTES: :[[@LINE-2]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:61: warning: 'end' iterator supplied where an output iterator is expected
  std::transform(I.begin(), I.begin(), J.rend(), K.end(), &dummyBinary<>);
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:40: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:50: warning: 'end' iterator supplied where an output iterator is expected
  std::transform(std::execution::par, I.begin(), I.begin(), J.end(), K.end(),
                 &dummyBinary<>);
  // CHECK-NOTES: :[[@LINE-2]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:61: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-4]]:70: warning: 'end' iterator supplied where an output iterator is expected

  std::rotate_copy(I.begin(), I.end(), RandomIIter, J.end());
  // CHECK-NOTES: :[[@LINE-1]]:31: warning: 'end' iterator passed as the middle iterator
  // CHECK-NOTES: :[[@LINE-2]]:53: warning: 'end' iterator supplied where an output iterator is expected
  std::rotate_copy(I.begin(), RandomIIter, I.end(), J.begin()); // OK
  std::rotate_copy(std::execution::par, I.begin(), I.end(), I.end(), J.begin());
  // CHECK-NOTES: :[[@LINE-1]]:52: warning: 'end' iterator passed as the middle iterator

  std::next(I.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: trying to increment past the end of a range
  std::prev(I.end()); // OK
  I.end() + 1;
  // CHECK-NOTES: :[[@LINE-1]]:11: warning: trying to increment past the end of a range
  I.end() - 1; // OK
  I.rbegin() - 1;
  // CHECK-NOTES: :[[@LINE-1]]:14: warning: trying to decrement before the start of a range
  I.rbegin() + 1; // OK
  I.rbegin().operator-(1);
  // CHECK-NOTES: :[[@LINE-1]]:14: warning: trying to decrement before the start of a range

  std::next(I.end(),
            -1); // Technically OK, getting the previous is a weird way.
  std::next(I.end(), 1);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: trying to increment past the end of a range
  std::next(I.begin(), 1); // OK
  std::next(I.begin(), -1);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: trying to decrement before the start of a range
  std::prev(I.end(), -1);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: trying to increment past the end of a range
  std::prev(I.end(), 1);    // OK
  std::prev(I.begin(), -1); // Technially OK, getting next in a weird way.
  std::prev(I.begin(), 1);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: trying to decrement before the start of a range

  I.end() - -1;
  // CHECK-NOTES: :[[@LINE-1]]:11: warning: trying to increment past the end of a range
  I.begin() + -1;
  // CHECK-NOTES: :[[@LINE-1]]:13: warning: trying to decrement before the start of a range

  std::make_reverse_iterator(I.begin()) - -1;
  // CHECK-NOTES: :[[@LINE-1]]:41: warning: trying to increment past the end of a range
  // CHECK-NOTES: :[[@LINE-2]]:3: note: 'make_reverse_iterator<int *>' changes 'begin' into an 'end' iterator
  std::make_reverse_iterator(I.end()) + -1;
  // CHECK-NOTES: :[[@LINE-1]]:39: warning: trying to decrement before the start of a range
  // CHECK-NOTES: :[[@LINE-2]]:3: note: 'make_reverse_iterator<int *>' changes 'end' into a 'begin' iterator

  std::set<int> Items{I.end(), I.end()};
  // CHECK-NOTES: :[[@LINE-1]]:23: warning: 'end' iterator supplied where a 'begin' iterator is expected [bugprone-incorrect-iterators]
  std::set<int> Other;
  Items.insert(Other.begin(), 5);
  // CHECK-NOTES: :[[@LINE-1]]:16: warning: 'insert' called with an iterator for a different container [bugprone-incorrect-iterators]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: container is specified here
  // CHECK-NOTES: :[[@LINE-3]]:16: note: different container provided here
  Items.insert(Other.begin(), Other.begin());
  // CHECK-NOTES: :[[@LINE-1]]:31: warning: 'begin' iterator supplied where an 'end' iterator is expected
}
