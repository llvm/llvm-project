// RUN: %check_clang_tidy %s performance-inefficient-vector-operation %t -- \
// RUN: -format-style=llvm \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: performance-inefficient-vector-operation.EnableProto, value: true}]}'

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

  template <class... Args> void emplace_back(Args &&... args);

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

namespace proto2 {
class MessageLite {};
class Message : public MessageLite {};
} // namespace proto2

class FooProto : public proto2::Message {
 public:
  int *add_x();  // repeated int x;
  void add_x(int x);
  void mutable_x();
  void mutable_y();
  int add_z() const; // optional int add_z;
};

class BarProto : public proto2::Message {
 public:
  int *add_x();
  void add_x(int x);
  void mutable_x();
  void mutable_y();
};

void f(std::vector<int>& t) {
  {
    std::vector<int> v0;
    // CHECK-FIXES: v0.reserve(10);
    for (int i = 0; i < 10; ++i)
      v0.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called inside a loop; consider pre-allocating the container capacity before the loop
  }
  {
    std::vector<int> v1;
    // CHECK-FIXES: v1.reserve(10);
    for (int i = 0; i < 10; i++)
      v1.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v2;
    // CHECK-FIXES: v2.reserve(10);
    for (int i = 0; i < 10; ++i)
      v2.push_back(0);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v3;
    // CHECK-FIXES: v3.reserve(5);
    for (int i = 0; i < 5; ++i) {
      v3.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
    // CHECK-FIXES-NOT: v3.reserve(10);
    for (int i = 0; i < 10; ++i) {
      // No fix for this loop as we encounter the prior loops.
      v3.push_back(i);
    }
  }
  {
    std::vector<int> v4;
    std::vector<int> v5;
    v5.reserve(3);
    // CHECK-FIXES: v4.reserve(10);
    for (int i = 0; i < 10; ++i)
      v4.push_back(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v6;
    // CHECK-FIXES: v6.reserve(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
      v6.push_back(t[i]);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<int> v7;
    // CHECK-FIXES: v7.reserve(t.size() - 1);
    for (std::size_t i = 0; i < t.size() - 1; ++i) {
      v7.push_back(t[i]);
    } // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
  }
  {
    std::vector<int> v8;
    // CHECK-FIXES: v8.reserve(t.size());
    for (const auto &e : t) {
      v8.push_back(e);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<int> v9;
    // CHECK-FIXES: v9.reserve(t.size());
    for (const auto &e : t) {
      v9.push_back(Op(e));
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<Foo> v10;
    // CHECK-FIXES: v10.reserve(t.size());
    for (const auto &e : t) {
      v10.push_back(Foo(e));
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<Bar> v11;
    // CHECK-FIXES: v11.reserve(t.size());
    for (const auto &e : t) {
      v11.push_back(e);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called
    }
  }
  {
    std::vector<Foo> v12;
    // CHECK-FIXES: v12.reserve(t.size());
    for (const auto &e : t) {
      v12.emplace_back(e);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'emplace_back' is called
    }
  }

  {
    FooProto foo;
    // CHECK-FIXES: foo.mutable_x()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_x(i);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'add_x' is called inside a loop; consider pre-allocating the container capacity before the loop
    }
  }

  // ---- Non-fixed Cases ----
  {
    std::vector<int> z0;
    z0.reserve(20);
    // CHECK-FIXES-NOT: z0.reserve(10);
    // There is a "reserve" call already.
    for (int i = 0; i < 10; ++i) {
      z0.push_back(i);
    }
  }
  {
    std::vector<int> z1;
    z1.reserve(5);
    // CHECK-FIXES-NOT: z1.reserve(10);
    // There is a "reserve" call already.
    for (int i = 0; i < 10; ++i) {
      z1.push_back(i);
    }
  }
  {
    std::vector<int> z2;
    z2.resize(5);
    // CHECK-FIXES-NOT: z2.reserve(10);
    // There is a ref usage of v before the loop.
    for (int i = 0; i < 10; ++i) {
      z2.push_back(i);
    }
  }
  {
    std::vector<int> z3;
    z3.push_back(0);
    // CHECK-FIXES-NOT: z3.reserve(10);
    // There is a ref usage of v before the loop.
    for (int i = 0; i < 10; ++i) {
      z3.push_back(i);
    }
  }
  {
    std::vector<int> z4;
    f(z4);
    // CHECK-FIXES-NOT: z4.reserve(10);
    // There is a ref usage of z4 before the loop.
    for (int i = 0; i < 10; ++i) {
      z4.push_back(i);
    }
  }
  {
    std::vector<int> z5(20);
    // CHECK-FIXES-NOT: z5.reserve(10);
    // z5 is not constructed with default constructor.
    for (int i = 0; i < 10; ++i) {
      z5.push_back(i);
    }
  }
  {
    std::vector<int> z6;
    // CHECK-FIXES-NOT: z6.reserve(10);
    // For-loop is not started with 0.
    for (int i = 1; i < 10; ++i) {
      z6.push_back(i);
    }
  }
  {
    std::vector<int> z7;
    // CHECK-FIXES-NOT: z7.reserve(t.size());
    // z7 isn't referenced in for-loop body.
    for (std::size_t i = 0; i < t.size(); ++i) {
      t.push_back(i);
    }
  }
  {
    std::vector<int> z8;
    int k;
    // CHECK-FIXES-NOT: z8.reserve(10);
    // For-loop isn't a fixable loop.
    for (std::size_t i = 0; k < 10; ++i) {
      z8.push_back(t[i]);
    }
  }
  {
    std::vector<int> z9;
    // CHECK-FIXES-NOT: z9.reserve(i + 1);
    // The loop end expression refers to the loop variable i.
    for (int i = 0; i < i + 1; i++)
      z9.push_back(i);
  }
  {
    std::vector<int> z10;
    int k;
    // CHECK-FIXES-NOT: z10.reserve(10);
    // For-loop isn't a fixable loop.
    for (std::size_t i = 0; i < 10; ++k) {
      z10.push_back(t[i]);
    }
  }
  {
    std::vector<int> z11;
    // initializer_list should not trigger the check.
    for (int e : {1, 2, 3, 4, 5}) {
      z11.push_back(e);
    }
  }
  {
    std::vector<int> z12;
    std::vector<int>* z13 = &t;
    // We only support detecting the range init expression which references
    // container directly.
    // Complex range init expressions like `*z13` is not supported.
    for (const auto &e : *z13) {
      z12.push_back(e);
    }
  }

  {
    FooProto foo;
    foo.mutable_x();
    // CHECK-FIXES-NOT: foo.mutable_x()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_x(i);
    }
  }
  {
    FooProto foo;
    // CHECK-FIXES-NOT: foo.mutable_x()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_x(i);
      foo.add_x(i);
    }
  }
  {
    FooProto foo;
    // CHECK-FIXES-NOT: foo.mutable_x()->Reserve(5);
    foo.add_x(-1);
    for (int i = 0; i < 5; i++) {
      foo.add_x(i);
    }
  }
  {
    FooProto foo;
    BarProto bar;
    bar.mutable_x();
    // CHECK-FIXES-NOT: foo.mutable_x()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_x();
      bar.add_x();
    }
  }
  {
    FooProto foo;
    foo.mutable_y();
    // CHECK-FIXES-NOT: foo.mutable_x()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_x(i);
    }
  }
  {
    FooProto foo;
    // CHECK-FIXES-NOT: foo.mutable_z()->Reserve(5);
    for (int i = 0; i < 5; i++) {
      foo.add_z();
    }
  }
}
