// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  char* p = 0; 
  template<class T> T g(T x = &p) { return x; }
  template int g<int>(int);	// OK even though &p isn't an int.
}

// Don't impose access restrictions on explicit instantiations.
namespace test1 {
  class A {
    class Private {};
  public:
    typedef Private Public;
  };

  template <class T> class Temp {
    static Temp<A::Public> make() { return Temp<A::Public>(); }
  };
  template class Temp<A::Private>;

  // FIXME: this ought to be an error, but it isn't because Sema is
  // silently failing to create a declaration for the explicit
  // instantiation.
  template class Temp<A::Private> Temp<int>::make();
}

// Don't impose access restrictions on explicit specializations,
// either.  This goes here because it's an extension of the rule for
// explicit instantiations and doesn't have any independent support.
namespace test2 {
  class A {
    class Private {}; // expected-note {{implicitly declared private here}}
  public:
    typedef Private Public;
  };

  template <class T> class Temp {
    static Temp<A::Public> make();
  };
  template <> class Temp<A::Private> {
  public:
    Temp(int x) {}
  };

  template <> class Temp<A::Private> Temp<int>::make() {
    return Temp<A::Public>(0);
  }

  template <>
  class Temp<char> {
    static Temp<A::Private> make() { // expected-error {{is a private member}}
      return Temp<A::Public>(0);
    }
  };
}
