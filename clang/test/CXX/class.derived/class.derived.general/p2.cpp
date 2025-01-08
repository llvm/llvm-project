// RUN: %clang_cc1 %s -fsyntax-only -verify

namespace CurrentInstantiation {
  template<typename T>
  struct A0 { // expected-note 6{{definition of 'A0<T>' is not complete until the closing '}'}}
    struct B0 : A0 { }; // expected-error {{base class has incomplete type}}

    template<typename U>
    struct B1 : A0 { }; // expected-error {{base class has incomplete type}}

    struct B2;

    template<typename U>
    struct B3;

    struct B4 { // expected-note 2{{definition of 'CurrentInstantiation::A0::B4' is not complete until the closing '}'}}
      struct C0 : A0, B4 { }; // expected-error 2{{base class has incomplete type}}

      template<typename V>
      struct C1 : A0, B4 { }; // expected-error 2{{base class has incomplete type}}

      struct C2;

      template<typename V>
      struct C3;
    };

    template<typename U>
    struct B5 { // expected-note 2{{definition of 'B5<U>' is not complete until the closing '}'}}
      struct C0 : A0, B5 { }; // expected-error 2{{base class has incomplete type}}

      template<typename V>
      struct C1 : A0, B5 { }; // expected-error 2{{base class has incomplete type}}

      struct C2;

      template<typename V>
      struct C3;
    };
  };

  template<typename T>
  struct A0<T>::B2 : A0 { };

  template<typename T>
  template<typename U>
  struct A0<T>::B3 : A0 { };

  template<typename T>
  struct A0<T>::B4::C2 : A0, B4 { };

  template<typename T>
  template<typename V>
  struct A0<T>::B4::C3 : A0, B4 { };

  template<typename T>
  template<typename U>
  struct A0<T>::B5<U>::C2 : A0, B5 { };

  template<typename T>
  template<typename U>
  template<typename V>
  struct A0<T>::B5<U>::C3 : A0, B5 { };

  template<typename T>
  struct A0<T*> { // expected-note 2{{definition of 'A0<type-parameter-0-0 *>' is not complete until the closing '}'}}
    struct B0 : A0 { }; // expected-error {{base class has incomplete type}}

    template<typename U>
    struct B1 : A0 { }; // expected-error {{base class has incomplete type}}

    struct B2;

    template<typename U>
    struct B3;
  };

  template<typename T>
  struct A0<T*>::B2 : A0 { };

  template<typename T>
  template<typename U>
  struct A0<T*>::B3 : A0 { };
} // namespace CurrentInstantiation

namespace MemberOfCurrentInstantiation {
  template<typename T>
  struct A0 {
    struct B : B { }; // expected-error {{base class has incomplete type}}
                      // expected-note@-1 {{definition of 'MemberOfCurrentInstantiation::A0::B' is not complete until the closing '}'}}

    template<typename U>
    struct C : C<U> { }; // expected-error {{base class has incomplete type}}
                         // expected-note@-1 {{definition of 'C<U>' is not complete until the closing '}'}}
  };

  template<typename T>
  struct A1 {
    struct B; // expected-note {{definition of 'MemberOfCurrentInstantiation::A1<long>::B' is not complete until the closing '}'}}

    struct C : B { }; // expected-error {{base class has incomplete type}}

    struct B : C { }; // expected-note {{in instantiation of member class 'MemberOfCurrentInstantiation::A1<long>::C' requested here}}
  };

  template struct A1<long>; // expected-note {{in instantiation of member class 'MemberOfCurrentInstantiation::A1<long>::B' requested here}}

  template<>
  struct A1<short>::B {
    static constexpr bool f() {
      return true;
    }
  };

  static_assert(A1<short>::C::f());
} // namespace MemberOfCurrentInstantiation
