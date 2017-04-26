// RUN: %check_clang_tidy %s performance-inefficient-vector-operation %t -- -format-style=llvm -- --std=c++11

namespace std {

typedef int size_t;

template<class E> class initializer_list {
public:
  using value_type = E;
  using reference = E&;
  using const_reference = const E&;
  using size_type = size_t;
  using iterator = const E*;
  using const_iterator = const E*;
  initializer_list();
  size_t size() const; // number of elements
  const E* begin() const; // first element
  const E* end() const; // one past the last element
};

// initializer list range access
template<class E> const E* begin(initializer_list<E> il);
template<class E> const E* end(initializer_list<E> il);

template <class T>
class vector {
 public:
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;

  explicit vector();
  explicit vector(size_type n);

  void push_back(const T& val);
  void reserve(size_t n);
  void resize(size_t n);

  size_t size();
  const_reference operator[] (size_type) const;
  reference operator[] (size_type);

  const_iterator begin() const;
  const_iterator end() const;
};
} // namespace std

class Foo {
 public:
  explicit Foo(int);
};

class Bar {
 public:
  Bar(int);
};

int Op(int);

void f(std::vector<int>& t) {
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(10);
    for (int i = 0; i < 10; ++i)
      v.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called inside a loop; consider pre-allocating the vector capacity before the loop
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(10);
    for (int i = 0; i < 10; i++)
      v.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(10);
    for (int i = 0; i < 10; ++i)
      v.push_back(0);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(5);
    for (int i = 0; i < 5; ++i) {
      v.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
    // CHECK-FIXES-NOT: v.reserve(10);
    for (int i = 0; i < 10; ++i) {
      // No fix for this loop as we encounter the prior loops.
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    std::vector<int> v2;
    v2.reserve(3);
    // CHECK-FIXES: v.reserve(10);
    for (int i = 0; i < 10; ++i)
      v.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
      v.push_back(t[i]);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(t.size() - 1);
    for (std::size_t i = 0; i < t.size() - 1; ++i) {
      v.push_back(t[i]);
    } // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(t.size());
    for (const auto &e : t) {
      v.push_back(e);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<int> v;
    // CHECK-FIXES: v.reserve(t.size());
    for (const auto &e : t) {
      v.push_back(Op(e));
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<Foo> v;
    // CHECK-FIXES: v.reserve(t.size());
    for (const auto &e : t) {
      v.push_back(Foo(e));
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<Bar> v;
    // CHECK-FIXES: v.reserve(t.size());
    for (const auto &e : t) {
      v.push_back(e);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  // ---- Non-fixed Cases ----
  {
    std::vector<int> v;
    v.reserve(20);
    // CHECK-FIXES-NOT: v.reserve(10);
    // There is a "reserve" call already.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    v.reserve(5);
    // CHECK-FIXES-NOT: v.reserve(10);
    // There is a "reserve" call already.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    v.resize(5);
    // CHECK-FIXES-NOT: v.reserve(10);
    // There is a ref usage of v before the loop.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    v.push_back(0);
    // CHECK-FIXES-NOT: v.reserve(10);
    // There is a ref usage of v before the loop.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    f(v);
    // CHECK-FIXES-NOT: v.reserve(10);
    // There is a ref usage of v before the loop.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v(20);
    // CHECK-FIXES-NOT: v.reserve(10);
    // v is not constructed with default constructor.
    for (int i = 0; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    // CHECK-FIXES-NOT: v.reserve(10);
    // For-loop is not started with 0.
    for (int i = 1; i < 10; ++i) {
      v.push_back(i);
    }
  }
  {
    std::vector<int> v;
    // CHECK-FIXES-NOT: v.reserve(t.size());
    // v isn't referenced in for-loop body.
    for (std::size_t i = 0; i < t.size(); ++i) {
      t.push_back(i);
    }
  }
  {
    std::vector<int> v;
    int k;
    // CHECK-FIXES-NOT: v.reserve(10);
    // For-loop isn't a fixable loop.
    for (std::size_t i = 0; k < 10; ++i) {
      v.push_back(t[i]);
    }
  }
  {
    std::vector<int> v;
    // CHECK-FIXES-NOT: v.reserve(i + 1);
    // The loop end expression refers to the loop variable i.
    for (int i = 0; i < i + 1; i++)
      v.push_back(i);
  }
  {
    std::vector<int> v;
    int k;
    // CHECK-FIXES-NOT: v.reserve(10);
    // For-loop isn't a fixable loop.
    for (std::size_t i = 0; i < 10; ++k) {
      v.push_back(t[i]);
    }
  }
  {
    std::vector<int> v;
    // initializer_list should not trigger the check.
    for (int e : {1, 2, 3, 4, 5}) {
      v.push_back(e);
    }
  }
  {
    std::vector<int> v;
    std::vector<int>* v2 = &t;
    // We only support detecting the range init expression which references
    // container directly.
    // Complex range init expressions like `*v2` is not supported.
    for (const auto &e : *v2) {
      v.push_back(e);
    }
  }
}
