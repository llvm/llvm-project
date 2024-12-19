// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// This is heavily affected by the speculative resolution applied to CWG2311
// So behaviour shown here is subject to change.

// expected-no-diagnostics

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

    constexpr size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };

  template<typename T>
  struct vector {
    size_t sz;
    constexpr vector() : sz(0) {}
    constexpr vector(initializer_list<T> ilist) : sz(ilist.size()) {}
    constexpr vector(const vector& other) : sz(other.sz) {}
    constexpr std::size_t size() const { return sz; }
  };
}

// https://github.com/llvm/llvm-project/pull/77768#issuecomment-1908062472
namespace Issue1 {
  struct A {
    constexpr A() {}
  };

  struct B {
    int called_ctor;
    constexpr explicit B(A) : called_ctor(0) {}
    constexpr explicit B(std::vector<A>) : called_ctor(1) {}
  };

  struct C {
    B b;
    constexpr C() : b({A()}) {}
  };

  static_assert(C().b.called_ctor == 0);
}

// https://github.com/llvm/llvm-project/pull/77768#issuecomment-1957171805
namespace Issue2 {
  struct A {
    constexpr A(int x_) {}
    constexpr A(const std::vector<A>& a) {}
  };

  void f() {
    constexpr std::vector<A> a{1,2};
    constexpr std::vector<A> b{a};
    // -> constexpr std::vector<A> b(std::initializer_list<A>{ A(a) });
    static_assert(b.size() == 1);
  }
}
