// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -DSTD1 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -DSTD2 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -DSTD3 %s

namespace std {

#ifdef STD1
template <typename T>
struct is_trivially_relocatable {
    static constexpr bool value = __builtin_is_cpp_trivially_relocatable(T);
};

template <typename T>
constexpr bool is_trivially_relocatable_v = __builtin_is_cpp_trivially_relocatable(T);
#endif

#ifdef STD2
template <typename T>
struct __details_is_trivially_relocatable {
    static constexpr bool value = __builtin_is_cpp_trivially_relocatable(T);
};

template <typename T>
using is_trivially_relocatable  = __details_is_trivially_relocatable<T>;

template <typename T>
constexpr bool is_trivially_relocatable_v = __builtin_is_cpp_trivially_relocatable(T);
#endif


#ifdef STD3
template< class T, T v >
struct integral_constant {
    static constexpr T value = v;
};

template< bool B >
using bool_constant = integral_constant<bool, B>;

template <typename T>
struct __details_is_trivially_relocatable : bool_constant<__builtin_is_cpp_trivially_relocatable(T)> {};

template <typename T>
using is_trivially_relocatable  = __details_is_trivially_relocatable<T>;

template <typename T>
constexpr bool is_trivially_relocatable_v = is_trivially_relocatable<T>::value;
#endif

}

static_assert(std::is_trivially_relocatable<int>::value);

static_assert(std::is_trivially_relocatable<int&>::value);
// expected-error-re@-1 {{static assertion failed due to requirement 'std::{{.*}}is_trivially_relocatable<int &>::value'}} \
// expected-note@-1 {{'int &' is not trivially relocatable}} \
// expected-note@-1 {{because it is a reference type}}
static_assert(std::is_trivially_relocatable_v<int&>);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_trivially_relocatable_v<int &>'}} \
// expected-note@-1 {{'int &' is not trivially relocatable}} \
// expected-note@-1 {{because it is a reference type}}

namespace test_namespace {
    using namespace std;
    static_assert(is_trivially_relocatable<int&>::value);
    // expected-error-re@-1 {{static assertion failed due to requirement '{{.*}}is_trivially_relocatable<int &>::value'}} \
    // expected-note@-1 {{'int &' is not trivially relocatable}} \
    // expected-note@-1 {{because it is a reference type}}
    static_assert(is_trivially_relocatable_v<int&>);
    // expected-error@-1 {{static assertion failed due to requirement 'is_trivially_relocatable_v<int &>'}} \
    // expected-note@-1 {{'int &' is not trivially relocatable}} \
    // expected-note@-1 {{because it is a reference type}}
}


namespace concepts {
template <typename T>
requires std::is_trivially_relocatable<T>::value void f();  // #cand1

template <typename T>
concept C = std::is_trivially_relocatable_v<T>; // #concept2

template <C T> void g();  // #cand2

void test() {
    f<int&>();
    // expected-error@-1 {{no matching function for call to 'f'}} \
    // expected-note@#cand1 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note-re@#cand1 {{because '{{.*}}is_trivially_relocatable<int &>::value' evaluated to false}} \
    // expected-note@#cand1 {{'int &' is not trivially relocatable}} \
    // expected-note@#cand1 {{because it is a reference type}}

    g<int&>();
    // expected-error@-1 {{no matching function for call to 'g'}} \
    // expected-note@#cand2 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note@#cand2 {{because 'int &' does not satisfy 'C'}} \
    // expected-note@#concept2 {{because 'std::is_trivially_relocatable_v<int &>' evaluated to false}} \
    // expected-note@#concept2 {{'int &' is not trivially relocatable}} \
    // expected-note@#concept2 {{because it is a reference type}}
}
}
