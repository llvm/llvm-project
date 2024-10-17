// RUN: %clang_cc1 -std=c++23 -Wno-unused %s -verify

namespace FoundNothing {
  template<typename T>
  void f0(T &t) {
    t.x<0;
    t.x<0>; // expected-error {{expected expression}}
    t.x<0>1;
  }

  template<typename T>
  struct A {
    void f1() {
      this->x<0; // expected-error {{no member named 'x' in 'A<T>'}}
      this->x<0>; // expected-error {{no member named 'x' in 'A<T>'}}
                  // expected-error@-1 {{expected expression}}
      this->x<0>1; // expected-error {{no member named 'x' in 'A<T>'}}
    }
  };
} // namespace FoundNothing

namespace FoundSingleNonTemplate {
  void f0();

  struct A0;

  template<typename T>
  void g0(T &t) {
    t.f0<0;
    t.f0<0>; // expected-error {{expected expression}}
    t.f0<0>1;

    t.A0<0;
    t.A0<0>; // expected-error {{expected expression}}
    t.A0<0>1;
  }

  template<typename T>
  struct B {
    void f1();

    struct A1; // expected-note 3{{member 'A1' declared here}}

    void g1() {
      this->f0<0; // expected-error {{no member named 'f0' in 'B<T>'}}
      this->f0<0>; // expected-error {{no member named 'f0' in 'B<T>'}}
                   // expected-error@-1 {{expected expression}}
      this->f0<0>1; // expected-error {{no member named 'f0' in 'B<T>'}}

      this->A0<0; // expected-error {{no member named 'A0' in 'B<T>'}}
      this->A0<0>; // expected-error {{no member named 'A0' in 'B<T>'}}
                   // expected-error@-1 {{expected expression}}
      this->A0<0>1; // expected-error {{no member named 'A0' in 'B<T>'}}

      this->f1<0; // expected-error {{reference to non-static member function must be called}}
      this->f1<0>; // expected-error {{reference to non-static member function must be called}}
                   // expected-error@-1 {{expected expression}}
      this->f1<0>1; // expected-error {{reference to non-static member function must be called}}

      this->A1<0; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
      this->A1<0>; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
                   // expected-error@-1 {{expected expression}}
      this->A1<0>1; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
    }
  };
} // namespace FoundSingleNonTemplate

namespace FoundSingleTemplate {
  template<int I>
  void f0();

  template<int I>
  struct A0;

  template<typename T>
  void g0(T &t) {
    t.f0<0;
    t.f0<0>; // expected-error {{expected expression}}
    t.f0<0>1;

    t.A0<0;
    t.A0<0>; // expected-error {{expected expression}}
    t.A0<0>1;
  }

  template<typename T>
  struct B {
    template<int I>
    void f1(); // expected-note 2{{possible target for call}}

    template<int I>
    struct A1; // expected-note 2{{member 'A1' declared here}}

    void g1() {
      this->f0<0; // expected-error {{no member named 'f0' in 'B<T>'}}
      this->f0<0>; // expected-error {{no member named 'f0' in 'B<T>'}}
      this->f0<0>1; // expected-error {{no member named 'f0' in 'B<T>'}}
                    // expected-error@-1 {{expected ';' after expression}}

      this->A0<0; // expected-error {{no member named 'A0' in 'B<T>'}}
      this->A0<0>; // expected-error {{no member named 'A0' in 'B<T>'}}
      this->A0<0>1; // expected-error {{no member named 'A0' in 'B<T>'}}
                    // expected-error@-1 {{expected ';' after expression}}


      this->f1<0; // expected-error {{expected '>'}}
                  // expected-note@-1 {{to match this '<'}}
      this->f1<0>; // expected-error {{reference to non-static member function must be called}}
      this->f1<0>1; // expected-error {{reference to non-static member function must be called}}
                    // expected-error@-1 {{expected ';' after expression}}

      this->A1<0; // expected-error {{expected '>'}}
                  // expected-note@-1 {{to match this '<'}}
      this->A1<0>; // expected-error {{cannot refer to member 'A1' in 'B<T>' with '->'}}
      this->A1<0>1; // expected-error {{cannot refer to member 'A1' in 'B<T>' with '->'}}
                    // expected-error@-1 {{expected ';' after expression}}
    }
  };
} // namespace FoundSingleTemplate

namespace FoundAmbiguousNonTemplate {
  inline namespace N {
    int f0;

    struct A0;
  } // namespace N

  void f0();

  struct A0;

  template<typename T>
  void g0(T &t) {
    t.f0<0;
    t.f0<0>; // expected-error {{expected expression}}
    t.f0<0>1;

    t.A0<0;
    t.A0<0>; // expected-error {{expected expression}}
    t.A0<0>1;
  }

  template<typename T>
  struct B {
    void f1();

    struct A1; // expected-note 3{{member 'A1' declared here}}

    void g1() {
      this->f0<0; // expected-error {{no member named 'f0' in 'B<T>'}}
      this->f0<0>; // expected-error {{no member named 'f0' in 'B<T>'}}
                   // expected-error@-1 {{expected expression}}
      this->f0<0>1; // expected-error {{no member named 'f0' in 'B<T>'}}

      this->A0<0; // expected-error {{no member named 'A0' in 'B<T>'}}
      this->A0<0>; // expected-error {{no member named 'A0' in 'B<T>'}}
                   // expected-error@-1 {{expected expression}}
      this->A0<0>1; // expected-error {{no member named 'A0' in 'B<T>'}}

      this->f1<0; // expected-error {{reference to non-static member function must be called}}
      this->f1<0>; // expected-error {{reference to non-static member function must be called}}
                   // expected-error@-1 {{expected expression}}
      this->f1<0>1; // expected-error {{reference to non-static member function must be called}}

      this->A1<0; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
      this->A1<0>; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
                   // expected-error@-1 {{expected expression}}
      this->A1<0>1; // expected-error {{cannot refer to type member 'A1' in 'B<T>' with '->'}}
    }
  };
} // namespace FoundAmbiguousNonTemplates

namespace FoundAmbiguousTemplate {
  inline namespace N {
    template<int I>
    int f0; // expected-note 3{{candidate found by name lookup is 'FoundAmbiguousTemplate::N::f0'}}

    template<int I>
    struct A0; // expected-note 3{{candidate found by name lookup is 'FoundAmbiguousTemplate::N::A0'}}
  } // namespace N

  template<int I>
  void f0(); // expected-note 3{{candidate found by name lookup is 'FoundAmbiguousTemplate::f0'}}

  template<int I>
  struct A0; // expected-note 3{{candidate found by name lookup is 'FoundAmbiguousTemplate::A0'}}

  template<typename T>
  void g0(T &t) {
    t.f0<0;
    t.f0<0>; // expected-error {{expected expression}}
    t.f0<0>1;

    t.A0<0;
    t.A0<0>; // expected-error {{expected expression}}
    t.A0<0>1;
  }

  template<typename T>
  struct B {
    template<int I>
    void f1(); // expected-note 2{{possible target for call}}

    template<int I>
    struct A1; // expected-note 2{{member 'A1' declared here}}

    void g1() {
      this->f0<0; // expected-error {{no member named 'f0' in 'B<T>'}}
                  // expected-error@-1 {{reference to 'f0' is ambiguous}}
      this->f0<0>; // expected-error {{no member named 'f0' in 'B<T>'}}
                   // expected-error@-1 {{reference to 'f0' is ambiguous}}
      this->f0<0>1; // expected-error {{no member named 'f0' in 'B<T>'}}
                    // expected-error@-1 {{expected ';' after expression}}
                    // expected-error@-2 {{reference to 'f0' is ambiguous}}

      this->A0<0; // expected-error {{no member named 'A0' in 'B<T>'}}
                  // expected-error@-1 {{reference to 'A0' is ambiguous}}
      this->A0<0>; // expected-error {{no member named 'A0' in 'B<T>'}}
                   // expected-error@-1 {{reference to 'A0' is ambiguous}}
      this->A0<0>1; // expected-error {{no member named 'A0' in 'B<T>'}}
                    // expected-error@-1 {{expected ';' after expression}}
                    // expected-error@-2 {{reference to 'A0' is ambiguous}}

      this->f1<0; // expected-error {{expected '>'}}
                  // expected-note@-1 {{to match this '<'}}
      this->f1<0>; // expected-error {{reference to non-static member function must be called}}
      this->f1<0>1; // expected-error {{reference to non-static member function must be called}}
                    // expected-error@-1 {{expected ';' after expression}}

      this->A1<0; // expected-error {{expected '>'}}
                  // expected-note@-1 {{to match this '<'}}
      this->A1<0>; // expected-error {{cannot refer to member 'A1' in 'B<T>' with '->'}}
      this->A1<0>1; // expected-error {{cannot refer to member 'A1' in 'B<T>' with '->'}}
                    // expected-error@-1 {{expected ';' after expression}}
    }
  };
} // namespace FoundAmbiguousTemplate
