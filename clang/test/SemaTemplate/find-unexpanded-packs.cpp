// RUN: %clang_cc1 -verify %s -std=c++17 -Wno-unused

template<typename ...Ts> void PackInsideTypedefDeclaration() {
  ([] {
    typedef Ts Type;
    (void)Type();
  }(), ...);
}
template void PackInsideTypedefDeclaration<>();
template void PackInsideTypedefDeclaration<int>();
template void PackInsideTypedefDeclaration<int, float>();

template<typename ...Ts> void PackInsideTypedefDeclarationInvalid() {
  [] { // expected-error {{contains unexpanded parameter pack 'Ts'}}
    typedef Ts Type;
    (void)Type();
  };

  ([] {
    typedef Ts Type;
    // A reference to a typedef containing an unexpanded pack does not
    // itself contain an unexpanded pack.
    f(Type()...); // expected-error {{does not contain any unexpanded}}
  }, ...);
}


template<typename ...Ts> void PackInsideAliasDeclaration() {
  ([] {
    using Type = Ts;
    (void)Type();
  }(), ...);
}
template void PackInsideAliasDeclaration<>();
template void PackInsideAliasDeclaration<int>();
template void PackInsideAliasDeclaration<int, float>();

template<typename ...Ts> void PackInsideAliasDeclarationInvalid() {
  [] { // expected-error {{contains unexpanded parameter pack 'Ts'}}
    using Type = Ts;
    (void)Type();
  };
  ([] {
    using Type = Ts;
    // A reference to an alias containing an unexpanded pack does not
    // itself contain an unexpanded pack.
    f(Type()...); // expected-error {{does not contain any unexpanded}}
  }, ...);
}


template<typename ...Ts> void PackInsideUsingDeclaration() {
  ([] {
    struct A {
      using Type = Ts;
    };
    struct B : A {
      using typename A::Type;
    };
    (void)typename B::Type();
  }(), ...);
}
template void PackInsideUsingDeclaration<>();
template void PackInsideUsingDeclaration<int>();
template void PackInsideUsingDeclaration<int, float>();

template<typename ...Ts> void PackInsideUsingDeclarationInvalid() {
  ([] {
    struct A {
      using Type = Ts;
    };
    struct B : A {
      using typename A::Type...; // expected-error {{does not contain any unexpanded}}
    };
  }(), ...);
}


template<typename ...Ts> void PackInsideVarDeclaration() {
  ([] {
    Ts ts;
    (void)ts;
  }, ...);
}
template void PackInsideVarDeclaration<>();
template void PackInsideVarDeclaration<int>();
template void PackInsideVarDeclaration<int, float>();

template<typename ...Ts> void PackInsideVarDeclarationInvalid() {
  [] { // expected-error {{contains unexpanded parameter pack 'Ts'}}
    Ts ts;
    (void)ts;
  };
}


template<typename ...Ts> void PackInsideFunctionDeclaration() {
  ([] {
    Ts ts(Ts);
    ts({});
  }, ...);
}
template void PackInsideFunctionDeclaration<>();
template void PackInsideFunctionDeclaration<int>();
template void PackInsideFunctionDeclaration<int, float>();

template<typename ...Ts> void PackInsideFunctionDeclarationInvalid() {
  [] { // expected-error {{contains unexpanded parameter pack 'Ts'}}
    Ts ts(Ts);
    ts({});
  };
}


template<typename ...Ts> void PackInsideLocalClass() {
  ([] {
    class Local {
      Ts ts;
    };
    Local l;
  }, ...);
}
template void PackInsideLocalClass<>();
template void PackInsideLocalClass<int>();
template void PackInsideLocalClass<int, float>();

template<typename ...Ts> void PackInsideLocalClassInvalid() {
  [] { // expected-error {{contains unexpanded parameter pack 'Ts'}}
    class Local {
      Ts ts;
    };
    Local l;
  };
}

template<typename T> using Int = int;
struct AClass {};
template<typename T> using Class = AClass;
template<typename ...Ts> void HiddenPack() {
  (Int<Ts>(), ...);
  (Int<Ts>{}, ...);
  (Class<Ts>(), ...);
  (Class<Ts>{}, ...);

  ([] {
   Int<Ts>();
   }, ...);
  ([] {
   Int<Ts>{};
   }, ...);
  ([] {
   Class<Ts>();
   }, ...);
  ([] {
   Class<Ts>{};
   }, ...);
}
template void HiddenPack<>();
template void HiddenPack<int>();
template void HiddenPack<int, float>();

template<typename ...Ts> void HiddenPackInvalid() {
  Int<Ts>(); // expected-error {{unexpanded}}
  Int<Ts>{}; // expected-error {{unexpanded}}
  Class<Ts>(); // expected-error {{unexpanded}}
  Class<Ts>{}; // expected-error {{unexpanded}}
}
