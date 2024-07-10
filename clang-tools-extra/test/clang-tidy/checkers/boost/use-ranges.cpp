// RUN: %check_clang_tidy -std=c++14 %s boost-use-ranges %t
// RUN: %check_clang_tidy -std=c++17 %s boost-use-ranges %t -check-suffixes=,CPP17

// CHECK-FIXES: #include <boost/range/algorithm/find.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/reverse.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/set_algorithm.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/equal.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/permutation.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/heap_algorithm.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/copy_if.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/is_sorted.hpp>
// CHECK-FIXES-CPP17: #include <boost/algorithm/cxx17/reduce.hpp>
// CHECK-FIXES: #include <boost/range/adaptor/reversed.hpp>
// CHECK-FIXES: #include <boost/range/numeric.hpp>

namespace std {

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  constexpr const_iterator begin() const;
  constexpr const_iterator end() const;
  constexpr const_iterator cbegin() const;
  constexpr const_iterator cend() const;
  constexpr iterator begin();
  constexpr iterator end();
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

} // namespace std

namespace boost {
namespace range_adl_barrier {
template <typename T> void *begin(T &);
template <typename T> void *end(T &);
template <typename T> void *const_begin(const T &);
template <typename T> void *const_end(const T &);
} // namespace range_adl_barrier
using namespace range_adl_barrier;

template <typename T> void *rbegin(T &);
template <typename T> void *rend(T &);

template <typename T> void *const_rbegin(T &);
template <typename T> void *const_rend(T &);
namespace algorithm {

template <class InputIterator, class T, class BinaryOperation>
T reduce(InputIterator first, InputIterator last, T init, BinaryOperation bOp) {
  return init;
}
} // namespace algorithm
} // namespace boost

bool returnTrue(int val) {
  return true;
}

void stdLib() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::find(I, 0);

  std::reverse(I.cbegin(), I.cend());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::reverse(I);

  std::includes(I.begin(), I.end(), std::begin(J), std::end(J));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::includes(I, J);

  std::equal(std::cbegin(I), std::cend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::equal(I, J);

  std::next_permutation(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::next_permutation(I);

  std::push_heap(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::push_heap(I);

  std::copy_if(I.begin(), I.end(), J.begin(), &returnTrue);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::algorithm::copy_if(I, J.begin(), &returnTrue);

  std::is_sorted_until(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::algorithm::is_sorted_until(I);

  std::reduce(I.begin(), I.end());
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I);

  std::reduce(I.begin(), I.end(), 2);
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I, 2);

  std::reduce(I.begin(), I.end(), 0, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I, 0, [](int a, int b){ return a + b; });

  std::equal(boost::rbegin(I), boost::rend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::equal(boost::adaptors::reverse(I), J);

  std::accumulate(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::accumulate(I, 0);
}

void boostLib() {
  std::vector<int> I;
  boost::algorithm::reduce(I.begin(), I.end(), 0, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 0, [](int a, int b){ return a + b; });

  boost::algorithm::reduce(boost::begin(I), boost::end(I), 1, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 1, [](int a, int b){ return a + b; });

  boost::algorithm::reduce(boost::const_begin(I), boost::const_end(I), 2, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 2, [](int a, int b){ return a + b; });
}
