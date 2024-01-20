// RUN: %check_clang_tidy -std=c++17 %s readability-redundant-inline-specifier %t
// RUN: %check_clang_tidy -std=c++17 -check-suffixes=,STRICT %s readability-redundant-inline-specifier %t -- -config="{CheckOptions: {readability-redundant-inline-specifier.StrictMode: 'true'}}"

template <typename T> inline T f()
// CHECK-MESSAGES-STRICT: :[[@LINE-1]]:23: warning: function 'f' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES-STRICT: template <typename T> T f()
{
    return T{};
}

template <> inline double f<double>() = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function 'f<double>' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: template <> double f<double>() = delete;

inline int g(float a)
{
    return static_cast<int>(a - 5.F);
}

inline int g(double) = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: function 'g' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: int g(double) = delete;

class C
{
  public:
    inline C& operator=(const C&) = delete;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'operator=' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: C& operator=(const C&) = delete;

    inline C(const C&) = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'C' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: C(const C&) = default;

    constexpr inline C& operator=(int a);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: function 'operator=' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr C& operator=(int a);

    inline C() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'C' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: C() {}

    constexpr inline C(int);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: function 'C' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr C(int);

    inline int Get42() const { return 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'Get42' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: int Get42() const { return 42; }

    static inline constexpr int C_STATIC = 42;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'C_STATIC' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: static constexpr int C_STATIC = 42;

    static constexpr int C_STATIC_2 = 42;
};

constexpr inline int Get42() { return 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: function 'Get42' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: constexpr int Get42() { return 42; }


static constexpr inline int NAMESPACE_STATIC = 42;

inline static int fn0(int i)
// CHECK-MESSAGES-STRICT: :[[@LINE-1]]:1: warning: function 'fn0' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES-STRICT: static int fn0(int i)
{
    return i - 1;
}

static constexpr inline int fn1(int i)
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function 'fn1' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: static constexpr int fn1(int i)
{
    return i - 1;
}

namespace
{
    inline int fn2(int i)
    // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:5: warning: function 'fn2' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES-STRICT: int fn2(int i)
    {
        return i - 1;
    }

    inline constexpr int fn3(int i)
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'fn3' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr int fn3(int i)
    {
        return i - 1;
    }

    inline constexpr int MY_CONSTEXPR_VAR = 42;
    // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:5: warning: variable 'MY_CONSTEXPR_VAR' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES-STRICT: constexpr int MY_CONSTEXPR_VAR = 42;
}

namespace ns
{
    inline int fn4(int i)
    {
        return i - 1;
    }

    inline constexpr int fn5(int i)
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'fn5' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr int fn5(int i)
    {
        return i - 1;
    }
}

auto fn6 = [](){};

template <typename T> inline T fn7();

template <typename T> T fn7()
{
    return T{};
}

template <typename T>  T fn8();

template <typename T> inline T fn8()
// CHECK-MESSAGES-STRICT: :[[@LINE-1]]:23: warning: function 'fn8' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES-STRICT: template <typename T> T fn8()
{
    return T{};
}

#define INLINE_MACRO() inline void fn9() { }
INLINE_MACRO()

#define INLINE_KW inline
INLINE_KW void fn10() { }
