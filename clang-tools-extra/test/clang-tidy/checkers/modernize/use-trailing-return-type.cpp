// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-use-trailing-return-type %t -- -- -fdeclspec -fexceptions -DCOMMAND_LINE_INT=int

namespace std {
    template <typename T>
    class vector;

    template <typename T, int N>
    class array;

    class string;

    template <typename T>
    class basic_ostream;

    using ostream = basic_ostream<char>;

    template <typename T>
    auto declval() -> T;

    template <typename... Ts>
    class tuple;
}

//
// Functions
//

int f();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f() -> int;
int ((f))();
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto ((f))() -> int;
int f(int);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f(int) -> int;
int f(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f(int arg) -> int;
int f(int arg1, int arg2, int arg3);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f(int arg1, int arg2, int arg3) -> int;
int f(int arg1, int arg2, int arg3, ...);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f(int arg1, int arg2, int arg3, ...) -> int;
template <typename T> int f(T t);
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: template <typename T> auto f(T t) -> int;

//
// Functions with formatting
//

int a1() { return 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a1() -> int { return 42; }
int a2() {
    return 42;
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a2() -> int {
int a3()
{
    return 42;
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a3() -> int
int a4(int   arg   )   ;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a4(int   arg   ) -> int   ;
int a5
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a5
(int arg);
// CHECK-FIXES: (int arg) -> int;
const
int
*
a7
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
()
// CHECK-FIXES: () -> const
// CHECK-FIXES: int
// CHECK-FIXES: *
;

int*a7(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a7(int arg) -> int*;
template<template <typename> class C>
C<int>a8(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto a8(int arg) -> C<int>;


//
// Functions with qualifiers and specifiers
//

inline int d1(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto d1(int arg) -> int;
extern "C" int d2(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: extern "C" auto d2(int arg) -> int;
inline int d3(int arg) noexcept(true);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto d3(int arg) noexcept(true) -> int;
inline int d4(int arg) try { return 0; } catch(...) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto d4(int arg) -> int try { return 0; } catch(...) { return 0; }
int d5(int arg) throw();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto d5(int arg) throw() -> int;
static int d6(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto d6(int arg) -> int;
int static d6(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto static d6(int arg) -> int;
unsigned static int d7(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto d7(int arg) -> unsigned int;
const long static int volatile constexpr unsigned inline long d8(int arg);
// CHECK-MESSAGES: :[[@LINE-1]]:63: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static constexpr inline auto d8(int arg) -> const long int volatile unsigned long;
int constexpr d9();
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto constexpr d9() -> int;
inline int constexpr d10();
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto constexpr d10() -> int;
unsigned constexpr int d11();
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: constexpr auto d11() -> unsigned int;
unsigned extern int d13();
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: extern auto d13() -> unsigned int;
int static& d14();
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto d14() -> int &;
class DDD {
    int friend unsigned m1();
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: friend auto m1() -> int unsigned;
    int friend unsigned m1() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: friend auto m1() -> int unsigned { return 0; }
    const long int friend volatile constexpr unsigned inline long m2();
// CHECK-MESSAGES: :[[@LINE-1]]:67: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: friend constexpr inline auto m2() -> const long int volatile unsigned long;
    int virtual unsigned m3();
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto m3() -> int unsigned;
    template <typename T>
    int friend unsigned m4();
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: friend auto m4() -> int unsigned;
};

//
// Functions in namespaces
//

namespace N {
    int e1();
}
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto e1() -> int;
int N::e1() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto N::e1() -> int { return 0; }

//
// Functions with unsupported return types
//
int (*e3())(double);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: int (*e3())(double);
struct A;
int A::* e5();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: int A::* e5();
int std::vector<std::string>::* e6();
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: int std::vector<std::string>::* e6();
int (std::vector<std::string>::*e7())(double);
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: int (std::vector<std::string>::*e7())(double);

//
// Functions with complex return types
//

inline volatile const std::vector<std::string> e2();
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto e2() -> volatile const std::vector<std::string>;
inline const std::vector<std::string> volatile e2();
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto e2() -> const std::vector<std::string> volatile;
inline std::vector<std::string> const volatile e2();
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: inline auto e2() -> std::vector<std::string> const volatile;
int* e8();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto e8() -> int*;
static const char* e9(void* user_data);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto e9(void* user_data) -> const char*;
static const char* const e10(void* user_data);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto e10(void* user_data) -> const char* const;
static const char** volatile * const & e11(void* user_data);
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto e11(void* user_data) -> const char** volatile * const &;
static const char* const * const * const e12(void* user_data);
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: static auto e12(void* user_data) -> const char* const * const * const;
struct A e13();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto e13() -> struct A;

//
// deduced return types
//

const auto ded1();
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto ded1() -> const auto;
const auto& ded2();
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto ded2() -> const auto&;

decltype(auto) ded3();
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto ded3() -> decltype(auto);


decltype(1 + 2) dec1() { return 1 + 2; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto dec1() -> decltype(1 + 2) { return 1 + 2; }
template <typename F, typename T>
decltype(std::declval<F>(std::declval<T>)) dec2(F f, T t) { return f(t); }
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto dec2(F f, T t) -> decltype(std::declval<F>(std::declval<T>)) { return f(t); }
template <typename T>
typename decltype(std::declval<T>())::value_type dec3();
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto dec3() -> typename decltype(std::declval<T>())::value_type;
template <typename T>
decltype(std::declval<T>())* dec4();
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto dec4() -> decltype(std::declval<T>())*;

//
// Methods
//

struct B {
    B& operator=(const B&);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto operator=(const B&) -> B&;

    double base1(int, bool b);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base1(int, bool b) -> double;

    virtual double base2(int, bool b) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base2(int, bool b) -> double { return 0; }

    virtual float base3() const = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base3() const -> float = 0;

    virtual float base4() volatile = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base4() volatile -> float = 0;

    double base5(int, bool b) &&;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base5(int, bool b) && -> double;

    double base6(int, bool b) const &&;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base6(int, bool b) const && -> double;

    double base7(int, bool b) const & = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base7(int, bool b) const & -> double = delete;

    double base8(int, bool b) const volatile & = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base8(int, bool b) const volatile & -> double = delete;

    virtual const char * base9() const noexcept { return ""; }
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base9() const noexcept -> const char * { return ""; }
};

double B::base1(int, bool b) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto B::base1(int, bool b) -> double { return 0; }

struct D : B {
    virtual double f1(int, bool b) final;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto f1(int, bool b) -> double final;

    virtual double base2(int, bool b) override;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base2(int, bool b) -> double override;

    virtual float base3() const override final { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: virtual auto base3() const -> float override final { return 0; }

    const char * base9() const noexcept override { return ""; }
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto base9() const noexcept -> const char * override { return ""; }

    int f2() __restrict;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f2() __restrict -> int;

    volatile int* __restrict f3() const __restrict noexcept;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto f3() const __restrict noexcept -> volatile int* __restrict;
};

//
// Functions with attributes
//

int g1() [[asdf]];
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto g1() -> int {{[[][[]}}asdf{{[]][]]}};
[[noreturn]] int g2();
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{[[][[]}}noreturn{{[]][]]}} auto g2() -> int;
int g2 [[noreturn]] ();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto g2 {{[[][[]}}noreturn{{[]][]]}} () -> int;
int unsigned g3() __attribute__((cdecl)); // FunctionTypeLoc is null.
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
int unsigned __attribute__((cdecl)) g3() ; // FunctionTypeLoc is null.
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
__attribute__((cdecl)) int unsigned g3() ; // FunctionTypeLoc is null.
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use a trailing return type for this function [modernize-use-trailing-return-type]

//
// Templates
//
template <typename Container>
[[maybe_unused]] typename Container::value_type const volatile&& t1(Container& C) noexcept;
// CHECK-MESSAGES: :[[@LINE-1]]:66: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{[[][[]}}maybe_unused{{[]][]]}} auto t1(Container& C) noexcept -> typename Container::value_type const volatile&&;
template <typename T>
class BB {
    using type = int;

    template <typename U>
    typename BB<U>::type m1();
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto m1() -> typename BB<U>::type;
};

//
// Macros
//

#define DWORD unsigned int
DWORD h1();
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h1() -> DWORD;
#define INT int
#define UNSIGNED unsigned
UNSIGNED INT h2();
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h2() -> UNSIGNED INT;
#define CONST const
CONST int h3();
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h3() -> CONST int;
#define ALWAYS_INLINE inline
#define DLL_EXPORT __declspec(dllexport)
ALWAYS_INLINE DLL_EXPORT int h4();
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: ALWAYS_INLINE DLL_EXPORT auto h4() -> int;
#define DEPRECATED __attribute__((deprecated))
int h5() DEPRECATED;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h5() -> int DEPRECATED;
int DEPRECATED h5();
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto DEPRECATED h5() -> int;
DEPRECATED int h5();
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: DEPRECATED auto h5() -> int;
[[noreturn]] [[nodiscard]] DEPRECATED DLL_EXPORT int h6 [[deprecated]] ();
// CHECK-MESSAGES: :[[@LINE-1]]:54: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{[[][[]}}noreturn{{[]][]]}} {{[[][[]}}nodiscard{{[]][]]}} DEPRECATED DLL_EXPORT auto h6 {{[[][[]}}deprecated{{[]][]]}} () -> int;
#define FUNCTION_NAME(a, b) a##b
int FUNCTION_NAME(foo, bar)();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto FUNCTION_NAME(foo, bar)() -> int;
#define DEFINE_FUNCTION_1(a, b) int a##b()
DEFINE_FUNCTION_1(foo, bar);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define DEFINE_FUNCTION_2 int foo(int arg);
DEFINE_FUNCTION_2
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define DLL_EXPORT_const __declspec(dllexport) const
DLL_EXPORT_const int h7();
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define DLL_EXPORT_CONST __declspec(dllexport) CONST
DLL_EXPORT_CONST int h7();
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a trailing return type for this function [modernize-use-trailing-return-type]

template <typename T>
using Real = T;
#define PRECISION float
Real<PRECISION> h8() { return 0.; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h8() -> Real<PRECISION> { return 0.; }

#define MAYBE_UNUSED_MACRO [[maybe_unused]]
template <typename Container>
MAYBE_UNUSED_MACRO typename Container::value_type const volatile** const h9(Container& C) noexcept;
// CHECK-MESSAGES: :[[@LINE-1]]:74: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: MAYBE_UNUSED_MACRO auto h9(Container& C) noexcept -> typename Container::value_type const volatile** const;

#define NOEXCEPT noexcept
int h9(int arg) NOEXCEPT;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h9(int arg) NOEXCEPT -> int;
#define STATIC_INT static int
STATIC_INT h10();
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define UNSIGNED_STATIC_INT unsigned static int
UNSIGNED_STATIC_INT h11();
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define STATIC static
unsigned STATIC int h11();
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: STATIC auto h11() -> unsigned int;
#define STATIC_CONSTEXPR static constexpr
unsigned STATIC_CONSTEXPR int h12();
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: STATIC_CONSTEXPR auto h12() -> unsigned int;
#define STATIC_CONSTEXPR_LONG static constexpr long
unsigned STATIC_CONSTEXPR_LONG int h13();
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
DEPRECATED const int& h14();
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: DEPRECATED auto h14() -> const int&;
DEPRECATED const long static volatile unsigned& h15();
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: DEPRECATED static auto h15() -> const long volatile unsigned&;
#define WRAP(x) x
WRAP(const) int& h16();
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
WRAP(CONST) int& h16();
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
#define CONCAT(a, b) a##b
CONCAT(con, st) int& h16();
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
DEPRECATED const UNSIGNED& h17();
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: DEPRECATED auto h17() -> const UNSIGNED&;
DEPRECATED CONST UNSIGNED STATIC& h18();
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: DEPRECATED STATIC auto h18() -> CONST UNSIGNED &;
#define CONST_CAT con##st
CONST_CAT int& h19();
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h19() -> CONST_CAT int&;
#define CONST_F_MACRO WRAP(CONST_CAT)
CONST_F_MACRO int& h19();
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h19() -> CONST_F_MACRO int&;
// Macro COMMAND_LINE_INT is defined on the command line via: -DCOMMAND_LINE_INT=int
const COMMAND_LINE_INT& h20();
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h20() -> const COMMAND_LINE_INT&;
decltype(COMMAND_LINE_INT{}) h21();
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto h21() -> decltype(COMMAND_LINE_INT{});

//
// Name collisions
//
struct Object { long long value; };

Object j1(unsigned Object) { return {Object * 2}; }
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: Object j1(unsigned Object) { return {Object * 2}; }
::Object j1(unsigned Object);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j1(unsigned Object) -> ::Object;
const Object& j2(unsigned a, int b, char Object, long l);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: const Object& j2(unsigned a, int b, char Object, long l);
const struct Object& j2(unsigned a, int b, char Object, long l);
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j2(unsigned a, int b, char Object, long l) -> const struct Object&;
std::vector<Object> j3(unsigned Object);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: std::vector<Object> j3(unsigned Object);
std::vector<const Object> j7(unsigned Object);
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: std::vector<const Object> j7(unsigned Object);
std::vector<Object> j4(unsigned vector);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j4(unsigned vector) -> std::vector<Object>;
std::vector<::Object> j4(unsigned vector);
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j4(unsigned vector) -> std::vector<::Object>;
std::vector<struct Object> j4(unsigned vector);
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j4(unsigned vector) -> std::vector<struct Object>;
std::vector<Object> j4(unsigned Vector);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto j4(unsigned Vector) -> std::vector<Object>;
using std::vector;
vector<Object> j5(unsigned vector);
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: vector<Object> j5(unsigned vector);
constexpr auto Size = 5;
std::array<int, Size> j6(unsigned Size);
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: std::array<int, Size> j6(unsigned Size);
std::array<decltype(Size), (Size * 2) + 1> j8(unsigned Size);
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: std::array<decltype(Size), (Size * 2) + 1> j8(unsigned Size);
using std::ostream;
std::tuple<int, std::string, ostream>& operator<<(ostream& ostream, float i);
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: std::tuple<int, std::string, ostream>& operator<<(ostream& ostream, float i);

class CC {
    int Object;
    struct Object m();
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto m() -> struct Object;
};
Object CC::m() { return {0}; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto CC::m() -> Object { return {0}; }
class DD : public CC {
    ::Object g();
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto g() -> ::Object;
};
Object DD::g() {
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: auto DD::g() -> Object {
    return {0};
}

//
// bug 44206, no rewrite should happen due to collision with parameter name
//

ostream& operator<<(ostream& ostream, int i);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: ostream& operator<<(ostream& ostream, int i);

//
// Samples which do not trigger the check
//

auto f() -> int;
auto f(int) -> int;
auto f(int arg) -> int;
auto f(int arg1, int arg2, int arg3) -> int;
auto f(int arg1, int arg2, int arg3, ...) -> int;
template <typename T> auto f(T t) -> int;

auto ff();

void c();
void c(int arg);
void c(int arg) { return; }

struct D2 : B {
    D2();
    virtual ~D2();

    virtual auto f1(int, bool b) -> double final;
    virtual auto base2(int, bool b) -> double override;
    virtual auto base3() const -> float override final { return 0;  }

    operator double();
};

