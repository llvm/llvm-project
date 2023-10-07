// RUN: %check_clang_tidy -std=c++20 %s modernize-use-std-numbers %t

// CHECK-FIXES: #include <numbers>

namespace bar {
    double sqrt(double Arg);
    float sqrt(float Arg);
    template <typename T>
    auto sqrt(T val) { return sqrt(static_cast<double>(val)); }

    static constexpr double e = 2.718281828459045235360287471352662497757247093;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double e = std::numbers::e;
}

double exp(double Arg);
double log(double Arg);
double log2(double Arg);
double log10(double Arg);

template<typename T>
void sink(T&&) { }

#define MY_PI 3.1415926
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer std::numbers math constant [modernize-use-std-numbers]
// CHECK-FIXES: #define MY_PI std::numbers::pi
#define MY_PI2 static_cast<float>(3.1415926)
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: prefer std::numbers math constant [modernize-use-std-numbers]
// CHECK-FIXES: #define MY_PI2 static_cast<float>(std::numbers::pi)

#define INV_SQRT3 1 / bar::sqrt(3)

void foo(){
    static constexpr double Pi = 3.1415926;
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Pi = std::numbers::pi;

    static constexpr double Euler = 2.7182818;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Euler = std::numbers::e;

    static constexpr double Phi = 1.6180339;
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Phi = std::numbers::phi;

    static constexpr double PiCopy = Pi;
    static constexpr double PiDefine = MY_PI;

    // not close enough to match value (DiffThreshold)
    static constexpr double Pi2 = 3.14;
    static constexpr double Euler2 = 2.71;
    static constexpr double Phi2 = 1.61;

    static constexpr double Pi3 = 3.1415926L;
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Pi3 = std::numbers::pi;

    static constexpr double Euler3 = 2.7182818L;
    // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Euler3 = std::numbers::e;

    static constexpr double Phi3 = 1.6180339L;
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double Phi3 = std::numbers::phi;

    static constexpr long double Pi4 = 3.1415926L;
    // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr long double Pi4 = std::numbers::pi_v<long double>;

    static constexpr long double Euler4 = 2.7182818L;
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr long double Euler4 = std::numbers::e_v<long double>;

    static constexpr long double Phi4 = 1.6180339L;
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr long double Phi4 = std::numbers::phi_v<long double>;

    static constexpr double InvPi = 1.0 / Pi;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr double InvPi = std::numbers::inv_pi;

    using my_float = const float;
    static constexpr my_float Actually2MyFloat = 2;
    bar::sqrt(Actually2MyFloat);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2_v<float>;

    constexpr static auto One = 1;
    constexpr static auto Two = 2;

    bar::sqrt(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    bar::sqrt(Two);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    bar::sqrt(2.0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    auto Not2 = 2;
    Not2 = 42;
    bar::sqrt(Not2);

    const auto Actually2 = 2;
    bar::sqrt(Actually2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    exp(1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::e;

    exp(One);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::e;

    exp(1.00000000000001);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::e;

    log2(exp(1));
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log2e;

    log2(Euler);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log2e;

    log2(bar::e);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log2e;

    auto log2e = 1.4426950;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: auto log2e = std::numbers::log2e;

    log10(exp(1));
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log10e;

    log10(Euler);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log10e;

    log10(bar::e);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log10e;

    auto log10e = .434294;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: auto log10e = std::numbers::log10e;

    auto egamma = 0.5772156 * 42;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: auto egamma = std::numbers::egamma * 42;

    sink(InvPi);

    sink(1 / Pi);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_pi);

    sink(1 / bar::sqrt(Pi));
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_sqrtpi);

    sink(1 / bar::sqrt(MY_PI));
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_sqrtpi);


    log(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::ln2;

    log(10);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::ln10;

    bar::sqrt(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    sink(1 / bar::sqrt(3));
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_sqrt3);

    bar::sqrt(3);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt3;

    auto phi = 1.6180339;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: auto phi = std::numbers::phi;

    sink(Phi);

    sink((42 + bar::sqrt(5)) / 2);

    sink((1 + bar::sqrt(5)) / 2);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::phi);

    sink((bar::sqrt(5.0F) + 1) / 2);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::phi_v<float>);
}



template <typename T>
void baz(){
    static constexpr T Pi = 3.1415926;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Pi = std::numbers::pi;

    static constexpr T Euler = 2.7182818;
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Euler = std::numbers::e;

    static constexpr T Phi = 1.6180339;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Phi = std::numbers::phi;

    static constexpr T PiCopy = Pi;
    static constexpr T PiDefine = MY_PI;

    // not close enough to match value (DiffThreshold)
    static constexpr T Pi2 = 3.14;
    static constexpr T Euler2 = 2.71;
    static constexpr T Phi2 = 1.61;

    static constexpr T Pi3 = 3.1415926L;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Pi3 = std::numbers::pi_v<long double>;

    static constexpr T Euler3 = 2.7182818L;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Euler3 = std::numbers::e_v<long double>;

    static constexpr T Phi3 = 1.6180339L;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: static constexpr T Phi3 = std::numbers::phi_v<long double>;

    using my_float = const float;
    static constexpr my_float Actually2MyFloat = 2;
    bar::sqrt(Actually2MyFloat);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2_v<float>;

    constexpr static T One = 1;
    constexpr static T Two = 2;

    bar::sqrt(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    bar::sqrt(Two);

    bar::sqrt(2.0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    T Not2 = 2;
    Not2 = 42;
    bar::sqrt(Not2);

    const T Actually2 = 2;
    bar::sqrt(Actually2);

    exp(1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::e;

    exp(One);

    exp(1.00000000000001);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::e;

    log2(exp(1));
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log2e;

    log2(Euler);

    log2(bar::e);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log2e;

    T log2e = 1.4426950;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: T log2e = std::numbers::log2e;

    log10(exp(1));
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log10e;

    log10(Euler);

    log10(bar::e);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::log10e;

    T log10e = .434294;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: T log10e = std::numbers::log10e;

    T egamma = 0.5772156 * 42;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: T egamma = std::numbers::egamma * 42;

    sink(1 / Pi);

    sink(1 / bar::sqrt(Pi));

    sink(1 / bar::sqrt(MY_PI));
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_sqrtpi);


    log(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::ln2;

    log(10);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::ln10;

    bar::sqrt(2);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt2;

    sink(1 / bar::sqrt(3));
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::inv_sqrt3);

    bar::sqrt(3);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: std::numbers::sqrt3;

    T phi = 1.6180339;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: T phi = std::numbers::phi;

    sink((42 + bar::sqrt(5)) / 2);

    sink((1 + bar::sqrt(5)) / 2);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::phi);

    sink((bar::sqrt(5.0F) + 1) / 2);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer std::numbers math constant [modernize-use-std-numbers]
    // CHECK-FIXES: sink(std::numbers::phi_v<float>);
}

template <typename T>
void foobar(){
    const T Two = 2;
    bar::sqrt(Two);
}
void use_foobar() {
    foobar<float>();
}
