// RUN: %clang_cc1 -fsyntax-only -verify %s

struct NonDefaultConstructible {
  NonDefaultConstructible(const NonDefaultConstructible&); // expected-note{{candidate constructor}}
};

template<typename T, typename U>
struct X {
  static T member;
};

template<typename T, typename U>
T X<T, U>::member; // expected-error{{no matching constructor}}

// Okay; this is a declaration, not a definition.
template<>
NonDefaultConstructible X<NonDefaultConstructible, long>::member;

NonDefaultConstructible &test(bool b) {
  return b? X<NonDefaultConstructible, int>::member // expected-note{{instantiation}}
          : X<NonDefaultConstructible, long>::member;
}

namespace rdar9422013 {
  template<int>
  struct X {
    struct Inner {
      static unsigned array[17];
    };
  };

  template<> unsigned X<1>::Inner::array[]; // okay
}
