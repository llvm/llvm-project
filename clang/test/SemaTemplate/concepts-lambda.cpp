// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify %s -triple powerpc64-ibm-aix

namespace GH57945 {
  template<typename T>
    concept c = true;

  template<typename>
    auto f = []() requires c<void> {
    };

  void g() {
      f<int>();
  };
}

namespace GH57945_2 {
  template<typename>
    concept c = true;

  template<typename T>
    auto f = [](auto... args) requires c<T>  {
    };

  template <typename T>
  auto f2 = [](auto... args)
    requires (sizeof...(args) > 0)
  {};

  void g() {
      f<void>();
      f2<void>(5.0);
  }
}

namespace GH57958 {
  template<class> concept C = true;
  template<int> constexpr bool v = [](C auto) { return true; }(0);
  int _ = v<0>;
}
namespace GH57958_2 {
  template<class> concept C = true;
  template<int> constexpr bool v = [](C auto...) { return true; }(0);
  int _ = v<0>;
}

namespace GH57971 {
  template<typename>
    concept any = true;

  template<typename>
    auto f = [](any auto) {
    };

  using function_ptr = void(*)(int);
  function_ptr ptr = f<void>;
}

// GH58368: A lambda defined in a concept requires we store
// the concept as a part of the lambda context.
namespace LambdaInConcept {
using size_t = unsigned long;

template<size_t...Ts>
struct IdxSeq{};

template <class T, class... Ts>
concept NotLike = true;

template <size_t, class... Ts>
struct AnyExcept {
  template <NotLike<Ts...> T> operator T&() const;
  template <NotLike<Ts...> T> operator T&&() const;
};

template <class T>
  concept ConstructibleWithN = (requires {
                                []<size_t I, size_t... Idxs>
                                (IdxSeq<I, Idxs...>)
                                requires requires { T{AnyExcept<I, T>{}}; }
                                { }
                                (IdxSeq<1,2,3>{});
    });

struct Foo {
  int i;
  double j;
  char k;
};

static_assert(ConstructibleWithN<Foo>);

}

// GH60642 reported an assert being hit, make sure we don't assert.
namespace GH60642 {
template<auto Q> concept C = requires { Q.template operator()<float>(); };
template<class> concept D = true;
static_assert(C<[]<D>{}>);  // ok
template<class> concept E = C<[]<D>{}>;
static_assert(E<int>);  // previously Asserted.

// ensure we properly diagnose when "D" is false.
namespace DIsFalse {
template<auto Q> concept C = requires { Q.template operator()<float>(); };
template<class> concept D = false;
static_assert(C<[]<D>{}>);
// expected-error@-1{{static assertion failed}}
// expected-note@-2{{does not satisfy 'C'}}
// expected-note@-5{{because 'Q.template operator()<float>()' would be invalid: no matching member function for call to 'operator()'}}
template<class> concept E = C<[]<D>{}>;
static_assert(E<int>);
// expected-error@-1{{static assertion failed}}
// expected-note@-2{{because 'int' does not satisfy 'E'}}
// expected-note@-4{{does not satisfy 'C'}}
// expected-note@-11{{because 'Q.template operator()<float>()' would be invalid: no matching member function for call to 'operator()'}}
}
}

namespace ReturnTypeRequirementInLambda {
template <typename T>
concept C1 = true;

template <class T>
concept test = [] {
  return requires(T t) {
    { t } -> C1;
  };
}();

static_assert(test<int>);

template <typename T>
concept C2 = true;
struct S1 {
  int f1() { return 1; }
};

void foo() {
  auto make_caller = []<auto member> {
    return [](S1 *ps) {
      if constexpr (requires {
                      { (ps->*member)() } -> C2;
                    })
        ;
    };
  };

  auto caller = make_caller.operator()<&S1::f1>();
}
} // namespace ReturnTypeRequirementInLambda

namespace GH73418 {
void foo() {
  int x;
  [&x](auto) {
    return [](auto y) {
      return [](auto obj, auto... params)
        requires requires {
          sizeof...(params);
          [](auto... pack) {
            return sizeof...(pack);
          }(params...);
        }
      { return false; }(y);
    }(x);
  }(x);
}
} // namespace GH73418
