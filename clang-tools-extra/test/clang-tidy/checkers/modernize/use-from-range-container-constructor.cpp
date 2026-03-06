// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-from-range-container-constructor %t

#include <stddef.h>
// CHECK-FIXES: #include <stddef.h>
// CHECK-FIXES: #include <ranges>

// Stubs of affected std containers and other utilities, since we can't include
// the necessary headers in this test.
namespace std {

struct from_range_t { explicit from_range_t() = default; };
inline constexpr from_range_t from_range{};

template <typename T>
struct stub_iterator {
    using iterator_category = void;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    T& operator*() const;
    stub_iterator& operator++();
    bool operator==(const stub_iterator&) const;
};

template <typename T>
class initializer_list {
  using value_type = T;
  const T *_M_array;
  size_t _M_len;
  inline constexpr initializer_list(const T *__a, size_t __l)
      : _M_array(__a), _M_len(__l) {}
 public:
  inline constexpr initializer_list() noexcept : _M_array(nullptr), _M_len(0) {}
  inline constexpr size_t size() const noexcept { return _M_len; }
  inline constexpr const T *begin() const noexcept { return _M_array; }
  inline constexpr const T *end() const noexcept { return _M_array + _M_len; }
};

template <typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
  pair(T1 f, T2 s) : first(f), second(s) {}
};

template <typename T, typename Alloc = void>
struct vector {
    using value_type = T;
    typedef stub_iterator<T> iterator;
    typedef stub_iterator<T> const_iterator;
    vector() = default;
    vector(initializer_list<T>) {}
    template <typename InputIt>
    vector(InputIt, InputIt) {}
    vector(from_range_t, auto&&) {}
    iterator begin(); iterator end();
    const_iterator begin() const; const_iterator end() const;
    const_iterator cbegin() const; const_iterator cend() const;
    iterator rbegin(); iterator rend();
    size_t size() const { return 0; }
    void push_back(T t) {}
};

template <typename T> struct deque : vector<T> { using vector<T>::vector; };
template <typename T> struct list : vector<T> { using vector<T>::vector; };
template <typename T> struct forward_list : vector<T> { using vector<T>::vector; };
template <typename T, typename Compare = void, typename Alloc = void>
struct set : vector<T> { using vector<T>::vector; };
template <typename K, typename V, typename Compare = void, typename Alloc = void>
struct map : vector<pair<K, V>> { using vector<pair<K, V>>::vector; };
template <typename T, typename Hash = void, typename KeyEqual = void, typename Alloc = void>
struct unordered_set : vector<T> { using vector<T>::vector; };
template <typename K, typename V, typename Hash = void, typename KeyEqual = void, typename Alloc = void>
struct unordered_map : vector<pair<K, V>> { using vector<pair<K, V>>::vector; };

template <typename T, typename Container = vector<T>>
struct priority_queue {
    using value_type = T;
    priority_queue(auto, auto) {}
    priority_queue(from_range_t, auto&&) {}
};
template <typename T> struct queue : priority_queue<T> { using priority_queue<T>::priority_queue; };
template <typename T> struct stack : priority_queue<T> { using priority_queue<T>::priority_queue; };

template <typename T> struct basic_string : vector<T> {
  using vector<T>::vector;
  basic_string(const char*) {}
};
using string = basic_string<char>;
struct string_view { string_view(const char*); string_view(const string&); };

template <typename T> struct greater {};
template <typename T> struct hash {};
template <> struct hash<int> { size_t operator()(int) const { return 0; } };

template <typename T> struct unique_ptr {
    T *operator->();
    T& operator*();
    T *get();
};
template <typename T> unique_ptr<T> make_unique();

template <typename T> struct shared_ptr {
    T *operator->();
    T& operator*();
    T *get();
};
template <typename T> shared_ptr<T> make_shared();

template <typename C> auto begin(C& c) { return c.begin(); }
template <typename C> auto end(C& c) { return c.end(); }

} // namespace std

static void testWarnsForAllContainerTypes() {
  std::vector<int> Ints = {1, 2, 3};
  std::vector<int> Vector(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> Vector(std::from_range, Ints);

  std::deque<int> Deque(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use std::from_range for container construction
  // CHECK-FIXES: std::deque<int> Deque(std::from_range, Ints);

  std::forward_list<int> ForwardList(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use std::from_range for container construction
  // CHECK-FIXES: std::forward_list<int> ForwardList(std::from_range, Ints);

  std::list<int> List(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use std::from_range for container construction
  // CHECK-FIXES: std::list<int> List(std::from_range, Ints);

  std::set<int> Set(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::from_range for container construction
  // CHECK-FIXES: std::set<int> Set(std::from_range, Ints);

  std::vector<std::pair<int, int>> IntPairs = {{1, 1}, {2, 2}, {3, 3}};
  std::map<int, int> Map(IntPairs.begin(), IntPairs.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::from_range for container construction
  // CHECK-FIXES: std::map<int, int> Map(std::from_range, IntPairs);

  std::unordered_set<int> Uset(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use std::from_range for container construction
  // CHECK-FIXES: std::unordered_set<int> Uset(std::from_range, Ints);

  std::unordered_map<int, int> Umap(IntPairs.begin(), IntPairs.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::from_range for container construction
  // CHECK-FIXES: std::unordered_map<int, int> Umap(std::from_range, IntPairs);

  std::priority_queue<int> PriorityQueue(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::from_range for container construction
  // CHECK-FIXES: std::priority_queue<int> PriorityQueue(std::from_range, Ints);

  std::queue<int> Queue(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use std::from_range for container construction
  // CHECK-FIXES: std::queue<int> Queue(std::from_range, Ints);

  std::stack<int> Stack(Ints.begin(), Ints.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use std::from_range for container construction
  // CHECK-FIXES: std::stack<int> Stack(std::from_range, Ints);

  std::vector<char> Chars = {'a'};
  std::string String(Chars.begin(), Chars.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use std::from_range for container construction
  // CHECK-FIXES: std::string String(std::from_range, Chars);
}

class Hashable {
 public:
  explicit Hashable(int Value) : Value(Value) {}
  bool operator==(const Hashable& rhs) const { return Value == rhs.Value; }
  bool operator<(const Hashable& rhs) const { return Value < rhs.Value; }
  int Value;
};

namespace std {

template <>
struct hash<Hashable> {
  size_t operator()(const Hashable& h) const { return 1u; }
};

}  // namespace std

static void testPreservesCustomHashesAndComparators() {
  struct PairHash {
    size_t operator()(const std::pair<int, int>& P) const { return 1; }
  };
  std::vector<std::pair<int, int>> Pairs = {{1, 1}, {2, 2}, {3, 3}};
  std::unordered_set<std::pair<int, int>, PairHash> Uset1(Pairs.begin(), Pairs.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:53: warning: use std::from_range for container construction
  // CHECK-FIXES: std::unordered_set<std::pair<int, int>, PairHash> Uset1(std::from_range, Pairs);

  std::vector<Hashable> Hashables = {{}};
  std::unordered_set<Hashable> Uset2(Hashables.begin(), Hashables.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::from_range for container construction
  // CHECK-FIXES: std::unordered_set<Hashable> Uset2(std::from_range, Hashables);

  std::set<std::pair<int, int>, std::greater<std::pair<int, int>>> Set(Pairs.begin(), Pairs.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:68: warning: use std::from_range for container construction
  // CHECK-FIXES: std::set<std::pair<int, int>, std::greater<std::pair<int, int>>> Set(std::from_range, Pairs);
}

static void testWarnsForAllExpressions() {
  struct HasVectorMember {
    explicit HasVectorMember(std::set<int> Set) : VectorMember(Set.begin(), Set.end()) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: use std::from_range for container construction [modernize-use-from-range-container-constructor]
    // CHECK-FIXES: explicit HasVectorMember(std::set<int> Set) : VectorMember(std::from_range, Set) {}
    std::vector<int> VectorMember;
  };

  auto F = [](std::set<int> SetParam) {
    return std::vector<int>(SetParam.begin(), SetParam.end());
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::from_range for container construction [modernize-use-from-range-container-constructor]
    // CHECK-FIXES: return std::vector<int>(std::from_range, SetParam);
  };

  std::vector<int> Vector;
  F(std::set<int>(Vector.begin(), Vector.end()));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::from_range for container construction [modernize-use-from-range-container-constructor]
  // CHECK-FIXES: F(std::set<int>(std::from_range, Vector));

  size_t Size = std::vector<int>(Vector.begin(), Vector.end()).size();
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::from_range for container construction
  // CHECK-FIXES: size_t Size = std::vector<int>(std::from_range, Vector).size();

}

static void testWarnsForAllValidIteratorStyles() {
  std::vector<int> Source = {1, 2};
  std::vector<int> V1(Source.begin(), Source.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V1(std::from_range, Source);

  std::vector<int> V2(Source.cbegin(), Source.cend());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V2(std::from_range, Source);

  std::vector<int> V3(std::begin(Source), std::end(Source));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V3(std::from_range, Source);

  // Note: rbegin() is not valid, see TestNegativeCases().
}

static void testDereferencesCorrectly() {
  auto UniquePtr = std::make_unique<std::vector<int>>();
  *UniquePtr = {1};

  std::vector<int> V1(UniquePtr->begin(), UniquePtr->end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V1(std::from_range, *UniquePtr);

  std::vector<int> V2(std::begin(*UniquePtr), std::end(*UniquePtr));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V2(std::from_range, *UniquePtr);

  std::vector<int> *RawPtr = UniquePtr.get();
  std::vector<int> V3(RawPtr->begin(), RawPtr->end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V3(std::from_range, *RawPtr);

  std::vector<int> Arr[2];
  std::vector<int> *PArr = &Arr[0];
  std::vector<int> VComplex((PArr + 1)->begin(), (PArr + 1)->end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> VComplex(std::from_range, *(PArr + 1));
}

static void testTypeConversions() {
  {
    std::set<std::string_view> Source = {"a"};
    std::vector<std::string> Dest(Source.begin(), Source.end());
    // Attempting to use std::from_range here would fail to compile, since
    // std::string_view needs to be explicitly converted to std::string.
  }
  {
    std::set<std::string> Source = {"a"};
    std::vector<std::string_view> Dest(Source.begin(), Source.end());
    // Here std::from_range would succeed - since the conversion from string to
    // string_view is implicit - but we choose not to warn, in order to keep
    // the tool check simple.
  }

  struct ImplicitlyConvertible;
  struct ExplicitlyConvertible {
    ExplicitlyConvertible() = default;
    ExplicitlyConvertible(const ImplicitlyConvertible&) {}
  };
  struct ImplicitlyConvertible {
    ImplicitlyConvertible() = default;
    explicit ImplicitlyConvertible(const ExplicitlyConvertible&) {}
  };
  {
    std::vector<ExplicitlyConvertible> Source = {{}};
    std::vector<ImplicitlyConvertible> Dest(Source.begin(), Source.end());
    // Attempting to use std::from_range here would fail to compile, since
    // an explicit conversion is required.
  }
  {
    std::vector<ImplicitlyConvertible> Source = {{}};
    std::vector<ExplicitlyConvertible> Dest(Source.begin(), Source.end());
    // Here std::from_range would succeed - since the conversion is implicit -
    // but we choose not to warn, so as to keep the tool check simple.
  }
}

static void testShouldNotWarn() {
  std::vector<int> S1 = {1};
  std::vector<int> S2 = {2};

  std::vector<int> V1(S1.begin(), S2.end());
  std::vector<int> V2(S1.rbegin(), S1.rend());

  struct NoFromRangeConstructor {
    NoFromRangeConstructor(std::vector<int>::iterator Begin, std::vector<int>::iterator End) {}
  };
  NoFromRangeConstructor V3(S1.begin(), S1.end());
}

static void testDifferentObjectsSameMember() {
  struct Data {
    std::vector<int> VectorMember;
  };
  Data D1, D2;
  D1.VectorMember = {1, 2};
  D2.VectorMember = {3, 4};

  // This should NOT warn. It's a valid (though weird) iterator pair.
  std::vector<int> V(D1.VectorMember.begin(), D2.VectorMember.end());
}

static void testCommentMidExpression() {
  auto Ptr = std::make_unique<std::vector<int>>();

  // Test with whitespace and comments
  std::vector<int> V(Ptr  /* comment */ ->  begin(), Ptr->end());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::from_range for container construction
  // CHECK-FIXES: std::vector<int> V(std::from_range, *Ptr);
}

static void testMapFromPairs() {
  // A vector of pairs, but the first element is NOT const.
  std::vector<std::pair<int, int>> Source = {{1, 10}};

  // std::map::value_type is std::pair<const int, int>.
  // The iterator constructor handles the conversion from pair<int, int>
  // to pair<const int, int> internally.
  std::map<int, int> M(Source.begin(), Source.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::from_range for container construction
  // CHECK-FIXES: std::map<int, int> M(std::from_range, Source);
}

static void testMapIncompatibility() {
    std::vector<std::pair<std::string_view, int>> Source = {};
    std::map<std::string, int> M(Source.begin(), Source.end());
}


static void testOperatorPrecedence(std::vector<int> *P1, std::vector<int> *P2, bool Cond) {
    std::vector<int> V((Cond ? P1 : P2)->begin(), (Cond ? P1 : P2)->end());
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::from_range for container construction
    // CHECK-FIXES: std::vector<int> V(std::from_range, *(Cond ? P1 : P2));
}

static void testNoDoubleDereference() {
    auto Ptr = std::make_shared<std::vector<int>>();
    std::vector<int> V((*Ptr).begin(), (*Ptr).end());
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::from_range for container construction
    // CHECK-FIXES: std::vector<int> V(std::from_range, *Ptr);

}
