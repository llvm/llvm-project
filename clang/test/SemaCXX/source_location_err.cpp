// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify -DTEST=1 %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify -DTEST=2 %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify -DTEST=3 %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify -DTEST=4 %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify -DTEST=5 %s

#if TEST == 1
auto test1a = __builtin_source_location(); // expected-error {{'std::source_location::__impl' was not found}}

namespace std {
inline namespace NS {
  struct source_location;
}
}

auto test1b = __builtin_source_location(); // expected-error {{'std::source_location::__impl' was not found}}

namespace std {
inline namespace NS {
  struct source_location {
    struct __impl;
  };
}
}
auto test1c = __builtin_source_location(); // expected-error {{'std::source_location::__impl' was not found}}

#elif TEST == 2
auto test2a = __builtin_source_location(); // expected-error {{'std::source_location::__impl' was not found}}

namespace std {
inline namespace NS {
struct source_location {
  struct __impl { int x; };
};
}
}
auto test2b = __builtin_source_location(); // expected-error {{'std::source_location::__impl' must be standard-layout and have only two 'const char *' fields '_M_file_name' and '_M_function_name', and two integral fields '_M_line' and '_M_column'}}

#elif TEST == 3
namespace std {
struct source_location {
  struct __impl {
    int other_member;
    char _M_line;
    const char *_M_file_name;
    char _M_column;
    const char *_M_function_name;
  };
};
}
auto test3 = __builtin_source_location(); // expected-error {{'std::source_location::__impl' must be standard-layout and have only two 'const char *' fields '_M_file_name' and '_M_function_name', and two integral fields '_M_line' and '_M_column'}}

#elif TEST == 4
namespace std {
struct source_location {
  struct parent {};
  struct __impl : public parent {
    char _M_line;
    const char *_M_file_name;
    char _M_column;
    const char *_M_function_name;
  };
};
}
auto test4 = __builtin_source_location(); // expected-error {{'std::source_location::__impl' must be standard-layout and have only two 'const char *' fields '_M_file_name' and '_M_function_name', and two integral fields '_M_line' and '_M_column'}}


#elif TEST == 5
namespace std {
struct source_location {
  struct __impl {
    signed char _M_line; // odd integral type to choose, but ok!
    const char *_M_file_name;
    signed char _M_column;
    const char *_M_function_name;
    static int other_member; // static members are OK
  };
  using BuiltinT = decltype(__builtin_source_location()); // OK.
};
}

// Verify that the address cannot be used as a non-type template argument.
template <auto X = __builtin_source_location()>
auto fn1() {return X;} // expected-note {{candidate template ignored: substitution failure: non-type template argument does not refer to any declaration}}
auto test5a = fn1<>(); // expected-error {{no matching function for call to 'fn1'}}

// (But using integer subobjects by value is okay.)
template <auto X = __builtin_source_location()->_M_column>
auto fn2() {return X;}
auto test5b = fn2<>();

// While it's not semantically required, for efficiency, we ensure that two
// source-locations with the same content will point to the same object. Given
// the odd definition of the struct used here (using 'signed char'), any
// line-number modulo 256 will thus have the same content, and be deduplicated.
#line 128
constexpr auto sl1 = __builtin_source_location();
#line 384
constexpr auto sl2 = __builtin_source_location();
constexpr auto sl3 = __builtin_source_location();
static_assert(sl1 == sl2);
static_assert(sl1 != sl3);
static_assert(sl1->_M_line == -128);

#endif
