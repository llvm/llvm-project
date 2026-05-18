// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wunreachable-code -verify=ref,both      %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wunreachable-code -verify=expected,both %s -fexperimental-new-constant-interpreter

namespace std {
  using size_t = decltype(sizeof(0));
}

namespace VoidCast {
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p);
  static_assert(q == nullptr);

  static_assert((delete (int*)(void*)new int, true));
  static_assert((delete[] (int*)(void*)new int[2], true));

  static_assert((delete (float*)(void*)new int, true)); // both-error {{not an integral constant expression}} \
                                                        // both-note {{cast from 'void *' is not allowed in a constant expression because the pointed object type 'int' is not similar to the target type 'float'}}

  static_assert((delete[] (float*)(void*)new int[2], true)); // both-error {{not an integral constant expression}} \
                                                             // both-note {{cast from 'void *' is not allowed in a constant expression because the pointed object type 'int' is not similar to the target type 'float'}}
}

namespace ReplaceableAlloc {
  struct F {
    static void* operator new(std::size_t n) {
      return nullptr; // both-warning {{should not return a null pointer}}
    }
  };

  constexpr F *createF() {
    return new F(); // both-note {{call to class-specific 'operator new'}}
  }

  constexpr bool foo() {
    F *f = createF(); // both-note {{in call to}}

    delete f;
    return true;
  }
  static_assert(foo()); // both-error {{not an integral constant expression}} \
                        // both-note {{in call to}}
}

constexpr int a = 12;
constexpr const int *b = &a;
constexpr int *f = (int*)(void*)b;
static_assert(*f == 12);

namespace ExplicitThisInBacktrace {
  struct S {
    constexpr void foo(this const S& self) {
      __builtin_abort(); // both-note {{not valid in a constant expression}}
    }
  };

  constexpr bool test() {
    S s;
    s.foo(); // both-note {{in call to}}
    return true;
  }

  static_assert(test()); // both-error {{not an integral constant expression}} \
                         // both-note {{in call to}}
}

namespace ConstexprUnknownNestedVariables {
  struct T { constexpr int a() const { return 42; } };
  constexpr const T& f(const T& t) noexcept { return t; }
  constexpr int f() {
      const T& range = f(T());
      return [&] consteval { return range.a(); }();
  }

  static_assert(f() == 42);
}

namespace ConstexprUnknownReference {

  struct expected {
    int val;
  };

  extern void __assert_fail();
  bool test() {
    expected e(5);

    const int &x = e.val;
    /// We used to get a warning for an always-true comparison.
    &(static_cast<const int&>(x)) == &e.val ? void() : __assert_fail();
    return true;
  }

}
