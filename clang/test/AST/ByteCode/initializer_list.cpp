// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -fms-extensions -verify=ref,both %s

// both-no-diagnostics

namespace std {
  typedef decltype(sizeof(int)) size_t;
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
    constexpr const _E* begin() const {return __begin_;}
    constexpr const _E* end()   const {return __begin_ + __size_;}
  };
}

class Thing {
public:
  int m = 12;
  constexpr Thing(int m) : m(m) {}
  constexpr bool operator==(const Thing& that) const {
    return this->m == that.m;
  }
};

constexpr bool is_contained(std::initializer_list<Thing> Set, const Thing &Element) {
   return (*Set.begin() == Element);
}

constexpr int foo() {
  const Thing a{12};
  const Thing b{14};
  return is_contained({a}, b);
}

static_assert(foo() == 0);
