// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++11 -fblocks -Wno-deprecated-builtins -fms-extensions -Wno-microsoft %s -Wno-c++17-extensions
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++14 -fblocks -Wno-deprecated-builtins -fms-extensions -Wno-microsoft %s -Wno-c++17-extensions
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++1z -fblocks -Wno-deprecated-builtins -fms-extensions -Wno-microsoft %s
// RUN: %clang_cc1 -x c -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu11 -fblocks -Wno-deprecated-builtins -fms-extensions -Wno-microsoft %s

#ifdef __cplusplus

// expected-no-diagnostics

using Int = int;

struct NonPOD { NonPOD(int); };
enum Enum { EV };
struct POD { Enum e; int i; float f; NonPOD* p; };
struct Derives : POD {};
using ClassType = Derives;

union Union { int i; float f; };

struct HasAnonymousUnion {
  union {
    int i;
    float f;
  };
};

struct FinalClass final {
};

template<typename T>
struct PotentiallyFinal { };

template<typename T>
struct PotentiallyFinal<T*> final { };

template<>
struct PotentiallyFinal<int> final { };

struct SealedClass sealed {
};

template<typename T>
struct PotentiallySealed { };

template<typename T>
struct PotentiallySealed<T*> sealed { };

template<>
struct PotentiallySealed<int> sealed { };

void is_final() {
  static_assert(__is_final(SealedClass));
  static_assert(__is_final(PotentiallySealed<float*>));
  static_assert(__is_final(PotentiallySealed<int>));

  static_assert(!__is_final(PotentiallyFinal<float>));
  static_assert(!__is_final(PotentiallySealed<float>));
}

void is_sealed()
{
  static_assert(__is_sealed(SealedClass));
  static_assert(__is_sealed(PotentiallySealed<float*>));
  static_assert(__is_sealed(PotentiallySealed<int>));
  static_assert(__is_sealed(FinalClass));
  static_assert(__is_sealed(PotentiallyFinal<float*>));
  static_assert(__is_sealed(PotentiallyFinal<int>));

  static_assert(!__is_sealed(int));
  static_assert(!__is_sealed(Union));
  static_assert(!__is_sealed(Int));
  static_assert(!__is_sealed(Int[10]));
  static_assert(!__is_sealed(Union[10]));
  static_assert(!__is_sealed(Derives));
  static_assert(!__is_sealed(ClassType));
  static_assert(!__is_sealed(const void));
  static_assert(!__is_sealed(Int[]));
  static_assert(!__is_sealed(HasAnonymousUnion));
  static_assert(!__is_sealed(PotentiallyFinal<float>));
  static_assert(!__is_sealed(PotentiallySealed<float>));
}
#else
struct s1 {};

void is_destructible()
{
  (void)__is_destructible(int);
  (void)__is_destructible(struct s1);
  (void)__is_destructible(struct s2); // expected-error{{incomplete type 'struct s2' used in type trait expression}}
  // expected-note@-1{{}}
}
#endif
