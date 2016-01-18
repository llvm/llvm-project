// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

namespace dr1550 { // dr1550: yes
  int f(bool b, int n) {
    return (b ? (throw 0) : n) + (b ? n : (throw 0));
  }
}

namespace dr1560 { // dr1560: 3.5
  void f(bool b, int n) {
    (b ? throw 0 : n) = (b ? n : throw 0) = 0;
  }
  class X { X(const X&); };
  const X &get();
  const X &x = true ? get() : throw 0;
}

#if __cplusplus >= 201103L
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
    : __begin_(__b), __size_(__s) {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };

  template < class _T1, class _T2 > struct pair { _T2 second; };

  template<typename T> struct basic_string {
    basic_string(const T* x) {}
    ~basic_string() {};
  };
  typedef basic_string<char> string;

} // std

namespace dr1589 {   // dr1589: 3.7 c++11
  // Ambiguous ranking of list-initialization sequences

  void f0(long, int=0);                 // Would makes selection of #0 ambiguous
  void f0(long);                        // #0
  void f0(std::initializer_list<int>);  // #00
  void g0() { f0({1L}); }               // chooses #00

  void f1(int, int=0);                    // Would make selection of #1 ambiguous
  void f1(int);                           // #1
  void f1(std::initializer_list<long>);   // #2
  void g1() { f1({42}); }                 // chooses #2

  void f2(std::pair<const char*, const char*>, int = 0); // Would makes selection of #3 ambiguous
  void f2(std::pair<const char*, const char*>); // #3
  void f2(std::initializer_list<std::string>);  // #4
  void g2() { f2({"foo","bar"}); }              // chooses #4

  namespace with_error {
    void f0(long);                        // #0    expected-note {{candidate function}}
    void f0(std::initializer_list<int>);  // #00   expected-note {{candidate function}}
    void f0(std::initializer_list<int>, int = 0);  // Makes selection of #00 ambiguous \
    // expected-note {{candidate function}}
    void g0() { f0({1L}); }                 // chooses #00    expected-error{{call to 'f0' is ambiguous}}

    void f1(int);                           // #1   expected-note {{candidate function}}
    void f1(std::initializer_list<long>);   // #2   expected-note {{candidate function}}
    void f1(std::initializer_list<long>, int = 0);   // Makes selection of #00 ambiguous \
    // expected-note {{candidate function}}
    void g1() { f1({42}); }                 // chooses #2   expected-error{{call to 'f1' is ambiguous}}

    void f2(std::pair<const char*, const char*>); // #3   TODO: expected- note {{candidate function}}
    void f2(std::initializer_list<std::string>);  // #4   expected-note {{candidate function}}
    void f2(std::initializer_list<std::string>, int = 0);   // Makes selection of #00 ambiguous \
    // expected-note {{candidate function}}
    void g2() { f2({"foo","bar"}); }        // chooses #4   expected-error{{call to 'f2' is ambiguous}}
  }

} // dr1589
#endif
