// RUN: %check_clang_tidy -check-suffix=ALL -std=c++20 %s modernize-use-std-numbers %t
// RUN: %check_clang_tidy -check-suffix=ALL,IMPRECISE -std=c++20 %s modernize-use-std-numbers %t -- -config="{CheckOptions: { modernize-use-std-numbers.DiffThreshold: 0.01 }}"

// CHECK-FIXES-ALL: #include <numbers>

namespace bar {
    double sqrt(double Arg);
    float sqrt(float Arg);
    template <typename T>
    auto sqrt(T val) { return sqrt(static_cast<double>(val)); }

    static constexpr double e = 2.718281828459045235360287471352662497757247093;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:33: warning: prefer 'std::numbers::e' to this literal, differs by '0.00e+00' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double e = std::numbers::e;
}

double exp(double Arg);
double log(double Arg);

double log2(double Arg);
float log2(float Arg);
template <typename T>
auto log2(T val) { return log2(static_cast<double>(val)); }

double log10(double Arg);

template<typename T>
void sink(T&&) { }

void floatSink(float) {}

#define MY_PI 3.1415926

#define INV_SQRT3 1 / bar::sqrt(3)
#define NOT_INV_SQRT3 1 / bar::sqrt(3) + 1

using my_double = double;
using my_float = float;

void foo(){
    static constexpr double Pi = 3.1415926;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:34: warning: prefer 'std::numbers::pi' to this literal, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Pi = std::numbers::pi;

    static constexpr double Euler = 2.7182818;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:37: warning: prefer 'std::numbers::e' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Euler = std::numbers::e;

    static constexpr double Phi = 1.6180339;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:35: warning: prefer 'std::numbers::phi' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Phi = std::numbers::phi;

    static constexpr double PiCopy = Pi;
    static constexpr double PiDefineFromMacro = MY_PI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:49: warning: prefer 'std::numbers::pi' to this macro, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double PiDefineFromMacro = std::numbers::pi;

    static constexpr double Pi2 = 3.14;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:35: warning: prefer 'std::numbers::pi' to this literal, differs by '1.59e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr double Pi2 = std::numbers::pi;
    static constexpr double Euler2 = 2.71;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:38: warning: prefer 'std::numbers::e' to this literal, differs by '8.28e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr double Euler2 = std::numbers::e;
    static constexpr double Phi2 = 1.61;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:36: warning: prefer 'std::numbers::phi' to this literal, differs by '8.03e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr double Phi2 = std::numbers::phi;

    static constexpr double Pi3 = 3.1415926L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:35: warning: prefer 'std::numbers::pi_v<long double>' to this literal, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Pi3 = std::numbers::pi_v<long double>;

    static constexpr double Euler3 = 2.7182818L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:38: warning: prefer 'std::numbers::e_v<long double>' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Euler3 = std::numbers::e_v<long double>;

    static constexpr double Phi3 = 1.6180339L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:36: warning: prefer 'std::numbers::phi_v<long double>' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double Phi3 = std::numbers::phi_v<long double>;

    static constexpr long double Pi4 = 3.1415926L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:40: warning: prefer 'std::numbers::pi_v<long double>' to this literal, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr long double Pi4 = std::numbers::pi_v<long double>;

    static constexpr long double Euler4 = 2.7182818L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:43: warning: prefer 'std::numbers::e_v<long double>' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr long double Euler4 = std::numbers::e_v<long double>;

    static constexpr long double Phi4 = 1.6180339L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:41: warning: prefer 'std::numbers::phi_v<long double>' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr long double Phi4 = std::numbers::phi_v<long double>;

    static constexpr my_double Euler5 = 2.7182818;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:41: warning: prefer 'std::numbers::e' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr my_double Euler5 = std::numbers::e;

    static constexpr my_float Euler6 = 2.7182818;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:40: warning: prefer 'std::numbers::e' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr my_float Euler6 = std::numbers::e;

    static constexpr int NotEuler7 = 2.7182818;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:38: warning: prefer 'std::numbers::e' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr int NotEuler7 = std::numbers::e;

    static constexpr double InvPi = 1.0 / Pi;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:37: warning: prefer 'std::numbers::inv_pi'  to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr double InvPi = std::numbers::inv_pi;

    static constexpr my_float Actually2MyFloat = 2;
    bar::sqrt(Actually2MyFloat);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2_v<float>'  to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2_v<float>;

    sink(MY_PI);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::pi' to this macro, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::pi);

    auto X = 42.0;
    auto Y = X * 3.14;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:18: warning: prefer 'std::numbers::pi' to this literal, differs by '1.59e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: auto Y = X * std::numbers::pi;

    constexpr static auto One = 1;
    constexpr static auto Two = 2;

    bar::sqrt(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    bar::sqrt(Two);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    bar::sqrt(2.0);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    auto Not2 = 2;
    Not2 = 42;
    bar::sqrt(Not2);

    const auto Actually2 = 2;
    bar::sqrt(Actually2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    exp(1);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::e;

    exp(One);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::e;

    exp(1.00000000000001);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::e;

    log2(exp(1));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:10: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    log2(Euler);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    log2(bar::e);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    log2(Euler5);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    log2(Euler6);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e_v<float>;

    log2(NotEuler7);

    auto log2e = 1.4426950;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:18: warning: prefer 'std::numbers::log2e' to this literal, differs by '4.09e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: auto log2e = std::numbers::log2e;

    floatSink(log2(Euler));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e);

    floatSink(static_cast<float>(log2(Euler)));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e_v<float>);

    floatSink(1.4426950);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e' to this literal, differs by '4.09e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e);

    floatSink(static_cast<float>(1.4426950));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e_v<float>' to this literal, differs by '4.09e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e_v<float>);

    floatSink(log2(static_cast<float>(Euler)));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e_v<float>);

    floatSink(static_cast<float>(log2(static_cast<float>(Euler))));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e_v<float>);

    floatSink(static_cast<float>(log2(static_cast<int>(Euler))));

    floatSink(static_cast<int>(log2(static_cast<float>(Euler))));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: prefer 'std::numbers::log2e_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(static_cast<int>(std::numbers::log2e_v<float>));

    floatSink(1.4426950F);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e_v<float>' to this literal, differs by '1.93e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e_v<float>);

    floatSink(static_cast<double>(1.4426950F));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e' to this literal, differs by '1.93e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(std::numbers::log2e);

    floatSink(static_cast<int>(1.4426950F));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: prefer 'std::numbers::log2e_v<float>' to this literal, differs by '1.93e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: floatSink(static_cast<int>(std::numbers::log2e_v<float>));

    log10(exp(1));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log10e' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:11: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log10e;

    log10(Euler);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log10e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log10e;

    log10(bar::e);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log10e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log10e;

    auto log10e = .434294;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:19: warning: prefer 'std::numbers::log10e' to this literal, differs by '4.82e-07' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: auto log10e = std::numbers::log10e;

    auto egamma = 0.5772156 * 42;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:19: warning: prefer 'std::numbers::egamma' to this literal, differs by '6.49e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: auto egamma = std::numbers::egamma * 42;

    sink(InvPi);

    sink(1 / Pi);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_pi' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_pi);

    sink(1 / bar::sqrt(Pi));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrtpi' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrtpi);

    sink(1 / bar::sqrt(MY_PI));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrtpi' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:24: warning: prefer 'std::numbers::pi' to this macro, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrtpi);

    log(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::ln2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::ln2;

    log(10);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::ln10' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::ln10;

    bar::sqrt(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    sink(1 / bar::sqrt(3));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:14: warning: prefer 'std::numbers::sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrt3);

    sink(INV_SQRT3);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrt3' to this macro [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrt3);

    sink(NOT_INV_SQRT3);

    const auto inv_sqrt3f = .577350269F;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:29: warning: prefer 'std::numbers::inv_sqrt3_v<float>' to this literal, differs by '1.04e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: const auto inv_sqrt3f = std::numbers::inv_sqrt3_v<float>;

    bar::sqrt(3);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt3;

    auto somePhi = 1.6180339;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:20: warning: prefer 'std::numbers::phi' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: auto somePhi = std::numbers::phi;

    sink(Phi);

    sink((42 + bar::sqrt(5)) / 2);

    sink((1 + bar::sqrt(5)) / 2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::phi' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::phi);

    sink((bar::sqrt(5.0F) + 1) / 2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::phi_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::phi_v<float>);
}



template <typename T>
void baz(){
    static constexpr T Pi = 3.1415926;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:29: warning: prefer 'std::numbers::pi' to this literal, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Pi = std::numbers::pi;

    static constexpr T Euler = 2.7182818;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: prefer 'std::numbers::e' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Euler = std::numbers::e;

    static constexpr T Phi = 1.6180339;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:30: warning: prefer 'std::numbers::phi' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Phi = std::numbers::phi;

    static constexpr T PiCopy = Pi;
    static constexpr T PiDefineFromMacro = MY_PI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:44: warning: prefer 'std::numbers::pi' to this macro, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T PiDefineFromMacro = std::numbers::pi;

    static constexpr T Pi2 = 3.14;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:30: warning: prefer 'std::numbers::pi' to this literal, differs by '1.59e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr T Pi2 = std::numbers::pi;
    static constexpr T Euler2 = 2.71;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:33: warning: prefer 'std::numbers::e' to this literal, differs by '8.28e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr T Euler2 = std::numbers::e;
    static constexpr T Phi2 = 1.61;
    // CHECK-MESSAGES-IMPRECISE: :[[@LINE-1]]:31: warning: prefer 'std::numbers::phi' to this literal, differs by '8.03e-03' [modernize-use-std-numbers]
    // CHECK-FIXES-IMPRECISE: static constexpr T Phi2 = std::numbers::phi;

    static constexpr T Pi3 = 3.1415926L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:30: warning: prefer 'std::numbers::pi_v<long double>' to this literal, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Pi3 = std::numbers::pi_v<long double>;

    static constexpr T Euler3 = 2.7182818L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:33: warning: prefer 'std::numbers::e_v<long double>' to this literal, differs by '2.85e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Euler3 = std::numbers::e_v<long double>;

    static constexpr T Phi3 = 1.6180339L;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: prefer 'std::numbers::phi_v<long double>' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: static constexpr T Phi3 = std::numbers::phi_v<long double>;

    static constexpr my_float Actually2MyFloat = 2;
    bar::sqrt(Actually2MyFloat);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2_v<float>;

    constexpr static T One = 1;
    constexpr static T Two = 2;

    bar::sqrt(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    bar::sqrt(Two);

    bar::sqrt(2.0);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    T Not2 = 2;
    Not2 = 42;
    bar::sqrt(Not2);

    const T Actually2 = 2;
    bar::sqrt(Actually2);

    exp(1);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::e;

    exp(One);

    exp(1.00000000000001);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::e;

    log2(exp(1));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:10: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    log2(Euler);

    log2(bar::e);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log2e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log2e;

    T log2e = 1.4426950;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: prefer 'std::numbers::log2e' to this literal, differs by '4.09e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: T log2e = std::numbers::log2e;

    log10(exp(1));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log10e' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:11: warning: prefer 'std::numbers::e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log10e;

    log10(Euler);

    log10(bar::e);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::log10e' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::log10e;

    T log10e = .434294;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:16: warning: prefer 'std::numbers::log10e' to this literal, differs by '4.82e-07' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: T log10e = std::numbers::log10e;

    T egamma = 0.5772156 * 42;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:16: warning: prefer 'std::numbers::egamma' to this literal, differs by '6.49e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: T egamma = std::numbers::egamma * 42;

    sink(1 / Pi);

    sink(1 / bar::sqrt(Pi));

    sink(1 / bar::sqrt(MY_PI));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrtpi' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:24: warning: prefer 'std::numbers::pi' to this macro, differs by '5.36e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrtpi);


    log(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::ln2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::ln2;

    log(10);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::ln10' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::ln10;

    bar::sqrt(2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt2' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt2;

    sink(1 / bar::sqrt(3));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::inv_sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-MESSAGES-ALL: :[[@LINE-2]]:14: warning: prefer 'std::numbers::sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::inv_sqrt3);

    bar::sqrt(3);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:5: warning: prefer 'std::numbers::sqrt3' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: std::numbers::sqrt3;

    T phi = 1.6180339;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: prefer 'std::numbers::phi' to this literal, differs by '8.87e-08' [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: T phi = std::numbers::phi;

    sink((42 + bar::sqrt(5)) / 2);

    sink((1 + bar::sqrt(5)) / 2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::phi' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::phi);

    sink((bar::sqrt(5.0F) + 1) / 2);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: prefer 'std::numbers::phi_v<float>' to this formula [modernize-use-std-numbers]
    // CHECK-FIXES-ALL: sink(std::numbers::phi_v<float>);
}

template <typename T>
void foobar(){
    const T Two = 2;
    bar::sqrt(Two);
}
void use_templates() {
    foobar<float>();
    foobar<double>();

    baz<float>();
    baz<double>();
}

#define BIG_MARCO                                                              \
  struct InvSqrt3 {                                                            \
    template <typename T> static T get() { return 1 / bar::sqrt(3); }          \
  }

BIG_MARCO;

void use_BIG_MACRO() {
InvSqrt3 f{};
f.get<float>();
f.get<double>();
}
