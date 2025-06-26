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

template <typename T>
struct is_trivially_copyable {
    static constexpr bool value = __is_trivially_copyable(T);
};

template <typename T>
constexpr bool is_trivially_copyable_v = __is_trivially_copyable(T);

template <typename T, typename U>
struct is_assignable {
    static constexpr bool value = __is_assignable(T, U);
};

template <typename T, typename U>
constexpr bool is_assignable_v = __is_assignable(T, U);

template <typename T>
struct is_empty {
    static constexpr bool value = __is_empty(T);
};
template <typename T>
constexpr bool is_empty_v = __is_empty(T);

template <typename T>
struct is_standard_layout {
static constexpr bool value = __is_standard_layout(T);
};
template <typename T>
constexpr bool is_standard_layout_v = __is_standard_layout(T);
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

template <typename T>
struct __details_is_trivially_copyable {
    static constexpr bool value = __is_trivially_copyable(T);
};

template <typename T>
using is_trivially_copyable  = __details_is_trivially_copyable<T>;

template <typename T>
constexpr bool is_trivially_copyable_v = __is_trivially_copyable(T);

template <typename T, typename U>
struct __details_is_assignable {
    static constexpr bool value = __is_assignable(T, U);
};

template <typename T, typename U>
using is_assignable = __details_is_assignable<T, U>;

template <typename T, typename U>
constexpr bool is_assignable_v = __is_assignable(T, U);

template <typename T>
struct __details_is_empty {
    static constexpr bool value = __is_empty(T);
};
template <typename T>
using is_empty  = __details_is_empty<T>;
template <typename T>
constexpr bool is_empty_v = __is_empty(T);

template <typename T>
struct __details_is_standard_layout {
static constexpr bool value = __is_standard_layout(T);


};
template <typename T>
using is_standard_layout = __details_is_standard_layout<T>;
template <typename T>
constexpr bool is_standard_layout_v = __is_standard_layout(T);
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

template <typename T>
struct __details_is_trivially_copyable : bool_constant<__is_trivially_copyable(T)> {};

template <typename T>
using is_trivially_copyable  = __details_is_trivially_copyable<T>;

template <typename T>
constexpr bool is_trivially_copyable_v = is_trivially_copyable<T>::value;

template <typename T, typename U>
struct __details_is_assignable : bool_constant<__is_assignable(T, U)> {};

template <typename T, typename U>
using is_assignable  = __details_is_assignable<T, U>;

template <typename T, typename U>
constexpr bool is_assignable_v = is_assignable<T, U>::value;

template <typename T>
struct __details_is_empty : bool_constant<__is_empty(T)> {};
template <typename T>
using is_empty  = __details_is_empty<T>;
template <typename T>
constexpr bool is_empty_v = is_empty<T>::value;

template <typename T>
struct __details_is_standard_layout : bool_constant<__is_standard_layout(T)> {};
template <typename T>
using is_standard_layout = __details_is_standard_layout<T>;
template <typename T>
constexpr bool is_standard_layout_v = is_standard_layout<T>::value;
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

static_assert(std::is_trivially_copyable<int>::value);

static_assert(std::is_trivially_copyable<int&>::value);
// expected-error-re@-1 {{static assertion failed due to requirement 'std::{{.*}}is_trivially_copyable<int &>::value'}} \
// expected-note@-1 {{'int &' is not trivially copyable}} \
// expected-note@-1 {{because it is a reference type}}
static_assert(std::is_trivially_copyable_v<int&>);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_trivially_copyable_v<int &>'}} \
// expected-note@-1 {{'int &' is not trivially copyable}} \
// expected-note@-1 {{because it is a reference type}}


 // Direct tests
 static_assert(std::is_standard_layout<int>::value);
 static_assert(std::is_standard_layout_v<int>);

 static_assert(std::is_standard_layout<int&>::value);
 // expected-error-re@-1 {{static assertion failed due to requirement 'std::{{.*}}is_standard_layout<int &>::value'}} \
 // expected-note@-1 {{'int &' is not standard-layout}} \
 // expected-note@-1 {{because it is a reference type}}

 static_assert(std::is_standard_layout_v<int&>);
 // expected-error@-1 {{static assertion failed due to requirement 'std::is_standard_layout_v<int &>'}} \
 // expected-note@-1 {{'int &' is not standard-layout}} \
 // expected-note@-1 {{because it is a reference type}}

static_assert(!std::is_empty<int>::value);

static_assert(std::is_empty<int&>::value);
// expected-error-re@-1 {{static assertion failed due to requirement 'std::{{.*}}is_empty<int &>::value'}} \
// expected-note@-1 {{'int &' is not empty}} \
// expected-note@-1 {{because it is a reference type}}
static_assert(std::is_empty_v<int&>);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_empty_v<int &>'}} \
// expected-note@-1 {{'int &' is not empty}} \
// expected-note@-1 {{because it is a reference type}}


static_assert(std::is_assignable<int&, int>::value);

static_assert(std::is_assignable<int&, void>::value);
// expected-error-re@-1 {{static assertion failed due to requirement 'std::{{.*}}is_assignable<int &, void>::value'}} \
// expected-error@-1 {{assigning to 'int' from incompatible type 'void'}}
static_assert(std::is_assignable_v<int&, void>);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_assignable_v<int &, void>'}} \
// expected-error@-1 {{assigning to 'int' from incompatible type 'void'}}

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

    static_assert(is_trivially_copyable<int&>::value);
    // expected-error-re@-1 {{static assertion failed due to requirement '{{.*}}is_trivially_copyable<int &>::value'}} \
    // expected-note@-1 {{'int &' is not trivially copyable}} \
    // expected-note@-1 {{because it is a reference type}}
    static_assert(is_trivially_copyable_v<int&>);
    // expected-error@-1 {{static assertion failed due to requirement 'is_trivially_copyable_v<int &>'}} \
    // expected-note@-1 {{'int &' is not trivially copyable}} \
    // expected-note@-1 {{because it is a reference type}}

    static_assert(is_standard_layout<int&>::value);
     // expected-error-re@-1 {{static assertion failed due to requirement '{{.*}}is_standard_layout<int &>::value'}} \
     // expected-note@-1 {{'int &' is not standard-layout}} \
     // expected-note@-1 {{because it is a reference type}}

     static_assert(is_standard_layout_v<int&>);
     // expected-error@-1 {{static assertion failed due to requirement 'is_standard_layout_v<int &>'}} \
     // expected-note@-1 {{'int &' is not standard-layout}} \
     // expected-note@-1 {{because it is a reference type}}

    static_assert(is_assignable<int&, void>::value);
    // expected-error-re@-1 {{static assertion failed due to requirement '{{.*}}is_assignable<int &, void>::value'}} \
    // expected-error@-1 {{assigning to 'int' from incompatible type 'void'}}
    static_assert(is_assignable_v<int&, void>);
    // expected-error@-1 {{static assertion failed due to requirement 'is_assignable_v<int &, void>'}} \
    // expected-error@-1 {{assigning to 'int' from incompatible type 'void'}}

    static_assert(is_empty<int&>::value);
    // expected-error-re@-1 {{static assertion failed due to requirement '{{.*}}is_empty<int &>::value'}} \
    // expected-note@-1 {{'int &' is not empty}} \
    // expected-note@-1 {{because it is a reference type}} 
    static_assert(is_empty_v<int&>);
    // expected-error@-1 {{static assertion failed due to requirement 'is_empty_v<int &>'}} \
    // expected-note@-1 {{'int &' is not empty}} \
    // expected-note@-1 {{because it is a reference type}}
}


namespace concepts {
template <typename T>
requires std::is_trivially_relocatable<T>::value void f();  // #cand1

template <typename T>
concept C = std::is_trivially_relocatable_v<T>; // #concept2

template <C T> void g();  // #cand2

template <typename T>
requires std::is_trivially_copyable<T>::value void f2();  // #cand3

template <typename T>
concept C2 = std::is_trivially_copyable_v<T>; // #concept4

template <C2 T> void g2();  // #cand4

template <typename T, typename U>
requires std::is_assignable<T, U>::value void f4();  // #cand7

template <typename T, typename U>
concept C4 = std::is_assignable_v<T, U>; // #concept8

template <C4<void> T> void g4();  // #cand8

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

    f2<int&>();
    // expected-error@-1 {{no matching function for call to 'f2'}} \
    // expected-note@#cand3 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note-re@#cand3 {{because '{{.*}}is_trivially_copyable<int &>::value' evaluated to false}} \
    // expected-note@#cand3 {{'int &' is not trivially copyable}} \
    // expected-note@#cand3 {{because it is a reference type}}

    g2<int&>();
    // expected-error@-1 {{no matching function for call to 'g2'}} \
    // expected-note@#cand4 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note@#cand4 {{because 'int &' does not satisfy 'C2'}} \
    // expected-note@#concept4 {{because 'std::is_trivially_copyable_v<int &>' evaluated to false}} \
    // expected-note@#concept4 {{'int &' is not trivially copyable}} \
    // expected-note@#concept4 {{because it is a reference type}}

    f4<int&, void>();
    // expected-error@-1 {{no matching function for call to 'f4'}} \
    // expected-note@#cand7 {{candidate template ignored: constraints not satisfied [with T = int &, U = void]}} \
    // expected-note-re@#cand7 {{because '{{.*}}is_assignable<int &, void>::value' evaluated to false}} \
    // expected-error@#cand7 {{assigning to 'int' from incompatible type 'void'}}

    g4<int&>();
    // expected-error@-1 {{no matching function for call to 'g4'}} \
    // expected-note@#cand8 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note@#cand8 {{because 'C4<int &, void>' evaluated to false}} \
    // expected-note@#concept8 {{because 'std::is_assignable_v<int &, void>' evaluated to false}} \
    // expected-error@#concept8 {{assigning to 'int' from incompatible type 'void'}}
}
}


namespace std {
template <typename T>
struct is_replaceable {
    static constexpr bool value = __builtin_is_replaceable(T);
};

template <typename T>
constexpr bool is_replaceable_v = __builtin_is_replaceable(T);

}

static_assert(std::is_replaceable<int&>::value);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_replaceable<int &>::value'}} \
// expected-note@-1 {{'int &' is not replaceable}} \
// expected-note@-1 {{because it is a reference type}}
static_assert(std::is_replaceable_v<int&>);
// expected-error@-1 {{static assertion failed due to requirement 'std::is_replaceable_v<int &>'}} \
// expected-note@-1 {{'int &' is not replaceable}} \
// expected-note@-1 {{because it is a reference type}}
