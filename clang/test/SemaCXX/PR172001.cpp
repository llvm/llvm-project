// RUN: %clang_cc1 -std=c++20 -verify=test1 -DTEST1 %s
// RUN: %clang_cc1 -std=c++20 -verify=test2 -DTEST2 %s
// RUN: %clang_cc1 -std=c++20 -verify=test3 -DTEST3 %s
// RUN: %clang_cc1 -std=c++20 -verify=test4 -DTEST4 %s
// RUN: %clang_cc1 -std=c++20 -verify=test5 -DTEST5 %s

namespace std {
#ifdef TEST1
  // Case 1: Malformed struct (static members are wrong type)
  // This triggered the original crash (GH170015).
  struct partial_ordering {
    static constexpr int equivalent = 0;
    static constexpr int less = -1;
    static constexpr int greater = 1;
    static constexpr int unordered = 2;
  };
#elif defined(TEST2)
  // Case 2: partial_ordering is a typedef to int
  using partial_ordering = int;
#elif defined(TEST3)
  // Case 3: partial_ordering is a forward declaration
  struct partial_ordering; // test3-note {{forward declaration of 'std::partial_ordering'}}
#elif defined(TEST4)
  // Case 4: partial_ordering is a template (Paranoia check)
  template <class> struct partial_ordering {
    static const partial_ordering less;
    static const partial_ordering equivalent;
    static const partial_ordering greater;
    static const partial_ordering unordered;
  };
#elif defined(TEST5)
  // Case 5: strong_ordering is a typedef to int (covers GH56571)
  using strong_ordering = int;
#endif
}

void f() {
#ifdef TEST5
  int a = 0, b = 0; // int <=> int requires std::strong_ordering
#else
  float a = 0.0f, b = 0.0f; // float <=> float requires std::partial_ordering
#endif

  auto res = a <=> b; 

  // test1-error@-2 {{standard library implementation of 'std::partial_ordering' is not supported; the type does not have the expected form}}
  // test2-error@-3 {{cannot use builtin operator '<=>' because type 'std::partial_ordering' was not found; include <compare>}}
  // test3-error@-4 {{incomplete type 'std::partial_ordering' where a complete type is required}}
  // test4-error@-5 {{cannot use builtin operator '<=>' because type 'std::partial_ordering' was not found; include <compare>}}
  // test5-error@-6 {{cannot use builtin operator '<=>' because type 'std::strong_ordering' was not found; include <compare>}}
}
