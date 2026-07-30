// Header for PCH test cxx-friends.cpp

class A {
  int x;
  friend class F;
};

namespace PR12585 {
  struct future_base {
    template<typename> class setter;
  };
  template<typename> class promise {
    // We used to inject this into future_base with no access specifier,
    // then crash during AST writing.
    template<typename> friend class future_base::setter;
    int k;
  };
}

namespace Lazy {
  struct S {
    friend void doNotDeserialize();
  };
}

// Reduced testcase from libc++'s <valarray>. Used to crash with modules
// enabled.
namespace std {

template <class T> struct valarray;

template <class T> struct valarray {
  valarray();
  template <class U> friend struct valarray;
  template <class U> friend U *begin(valarray<U> &v);
};

struct gslice {
  valarray<int> size;
  gslice() {}
};

}
