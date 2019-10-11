// RUN: %check_clang_tidy -std=c++11-or-later %s abseil-upgrade-duration-conversions %t -- -- -I%S/Inputs
// FIXME: Fix the checker to work in C++17 mode.

using int64_t = long long;

#include "absl/time/time.h"

template <typename T> struct ConvertibleTo {
  operator T() const;
};

template <typename T>
ConvertibleTo<T> operator+(ConvertibleTo<T>, ConvertibleTo<T>);

template <typename T>
ConvertibleTo<T> operator*(ConvertibleTo<T>, ConvertibleTo<T>);

void arithmeticOperatorBasicPositive() {
  absl::Duration d;
  d *= ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(ConvertibleTo<int64_t>());
  d /= ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d /= static_cast<int64_t>(ConvertibleTo<int64_t>());
  d = ConvertibleTo<int64_t>() * d;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = static_cast<int64_t>(ConvertibleTo<int64_t>()) * d;
  d = d * ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = d * static_cast<int64_t>(ConvertibleTo<int64_t>());
  d = d / ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = d / static_cast<int64_t>(ConvertibleTo<int64_t>());
  d.operator*=(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d.operator*=(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  d.operator/=(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d.operator/=(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  d = operator*(ConvertibleTo<int64_t>(), d);
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = operator*(static_cast<int64_t>(ConvertibleTo<int64_t>()), d);
  d = operator*(d, ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = operator*(d, static_cast<int64_t>(ConvertibleTo<int64_t>()));
  d = operator/(d, ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = operator/(d, static_cast<int64_t>(ConvertibleTo<int64_t>()));
  ConvertibleTo<int64_t> c;
  d *= (c + c) * c + c;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>((c + c) * c + c)
  d /= (c + c) * c + c;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d /= static_cast<int64_t>((c + c) * c + c)
  d = d * c * c;
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-MESSAGES: [[@LINE-2]]:15: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = d * static_cast<int64_t>(c) * static_cast<int64_t>(c)
  d = c * d * c;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-MESSAGES: [[@LINE-2]]:15: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = static_cast<int64_t>(c) * d * static_cast<int64_t>(c)
  d = d / c * c;
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-MESSAGES: [[@LINE-2]]:15: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = d / static_cast<int64_t>(c) * static_cast<int64_t>(c)
}

void arithmeticOperatorBasicNegative() {
  absl::Duration d;
  d *= char{1};
  d *= 1;
  d *= int64_t{1};
  d *= 1.0f;
  d *= 1.0;
  d *= 1.0l;
  d /= char{1};
  d /= 1;
  d /= int64_t{1};
  d /= 1.0f;
  d /= 1.0;
  d /= 1.0l;
  d = d * char{1};
  d = d * 1;
  d = d * int64_t{1};
  d = d * 1.0f;
  d = d * 1.0;
  d = d * 1.0l;
  d = char{1} * d;
  d = 1 * d;
  d = int64_t{1} * d;
  d = 1.0f * d;
  d = 1.0 * d;
  d = 1.0l * d;
  d = d / char{1};
  d = d / 1;
  d = d / int64_t{1};
  d = d / 1.0f;
  d = d / 1.0;
  d = d / 1.0l;

  d *= static_cast<int>(ConvertibleTo<int>());
  d *= (int)ConvertibleTo<int>();
  d *= int(ConvertibleTo<int>());
  d /= static_cast<int>(ConvertibleTo<int>());
  d /= (int)ConvertibleTo<int>();
  d /= int(ConvertibleTo<int>());
  d = static_cast<int>(ConvertibleTo<int>()) * d;
  d = (int)ConvertibleTo<int>() * d;
  d = int(ConvertibleTo<int>()) * d;
  d = d * static_cast<int>(ConvertibleTo<int>());
  d = d * (int)ConvertibleTo<int>();
  d = d * int(ConvertibleTo<int>());
  d = d / static_cast<int>(ConvertibleTo<int>());
  d = d / (int)ConvertibleTo<int>();
  d = d / int(ConvertibleTo<int>());

  d *= 1 + ConvertibleTo<int>();
  d /= 1 + ConvertibleTo<int>();
  d = (1 + ConvertibleTo<int>()) * d;
  d = d * (1 + ConvertibleTo<int>());
  d = d / (1 + ConvertibleTo<int>());
}

template <typename T> void templateForOpsSpecialization(T) {}
template <>
void templateForOpsSpecialization<absl::Duration>(absl::Duration d) {
  d *= ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(ConvertibleTo<int64_t>());
}

template <int N> void arithmeticNonTypeTemplateParamSpecialization() {
  absl::Duration d;
  d *= N;
}

template <> void arithmeticNonTypeTemplateParamSpecialization<5>() {
  absl::Duration d;
  d *= ConvertibleTo<int>();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(ConvertibleTo<int>());
}

template <typename T> void templateOpsFix() {
  absl::Duration d;
  d *= ConvertibleTo<int64_t>();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(ConvertibleTo<int64_t>());
}

template <typename T, typename U> void templateOpsWarnOnly(T t, U u) {
  t *= u;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  absl::Duration d;
  d *= u;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

template <typename T> struct TemplateTypeOpsWarnOnly {
  void memberA(T t) {
    d *= t;
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  }
  template <typename U, typename V> void memberB(U u, V v) {
    u *= v;
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
    d *= v;
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  }

  absl::Duration d;
};

template <typename T, typename U>
void templateOpsInstantiationBeforeDefinition(T t, U u);

void arithmeticOperatorsInTemplates() {
  templateForOpsSpecialization(5);
  templateForOpsSpecialization(absl::Duration());
  arithmeticNonTypeTemplateParamSpecialization<1>();
  arithmeticNonTypeTemplateParamSpecialization<5>();
  templateOpsFix<int>();
  templateOpsWarnOnly(absl::Duration(), ConvertibleTo<int>());
  templateOpsInstantiationBeforeDefinition(absl::Duration(),
                                           ConvertibleTo<int>());
  TemplateTypeOpsWarnOnly<ConvertibleTo<int>> t;
  t.memberA(ConvertibleTo<int>());
  t.memberB(absl::Duration(), ConvertibleTo<int>());
}

template <typename T, typename U>
void templateOpsInstantiationBeforeDefinition(T t, U u) {
  t *= u;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  absl::Duration d;
  d *= u;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

#define FUNCTION_MACRO(x) x
#define CONVERTIBLE_TMP ConvertibleTo<int>()
#define ONLY_WARN_INSIDE_MACRO_ARITHMETIC_OP d *= ConvertibleTo<int>()

#define T_OBJECT T()
#define T_CALL_EXPR d *= T()

template <typename T> void arithmeticTemplateAndMacro() {
  absl::Duration d;
  d *= T_OBJECT;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  d *= CONVERTIBLE_TMP;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(CONVERTIBLE_TMP);
  T_CALL_EXPR;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

#define TEMPLATE_MACRO(type)                                                   \
  template <typename T> void TemplateInMacro(T t) {                            \
    type d;                                                                    \
    d *= t;                                                                    \
  }

TEMPLATE_MACRO(absl::Duration)
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead

void arithmeticOperatorsInMacros() {
  absl::Duration d;
  d = FUNCTION_MACRO(d * ConvertibleTo<int>());
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d = FUNCTION_MACRO(d * static_cast<int64_t>(ConvertibleTo<int>()));
  d *= FUNCTION_MACRO(ConvertibleTo<int>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(FUNCTION_MACRO(ConvertibleTo<int>()));
  d *= CONVERTIBLE_TMP;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: d *= static_cast<int64_t>(CONVERTIBLE_TMP);
  ONLY_WARN_INSIDE_MACRO_ARITHMETIC_OP;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  arithmeticTemplateAndMacro<ConvertibleTo<int>>();
  TemplateInMacro(ConvertibleTo<int>());
}

void factoryFunctionPositive() {
  // User defined conversion:
  (void)absl::Nanoseconds(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  (void)absl::Microseconds(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Microseconds(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  (void)absl::Milliseconds(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Milliseconds(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  (void)absl::Seconds(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Seconds(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  (void)absl::Minutes(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Minutes(static_cast<int64_t>(ConvertibleTo<int64_t>()));
  (void)absl::Hours(ConvertibleTo<int64_t>());
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Hours(static_cast<int64_t>(ConvertibleTo<int64_t>()));

  // User defined conversion to integral type, followed by built-in conversion:
  (void)absl::Nanoseconds(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(ConvertibleTo<char>()));
  (void)absl::Microseconds(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Microseconds(static_cast<int64_t>(ConvertibleTo<char>()));
  (void)absl::Milliseconds(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Milliseconds(static_cast<int64_t>(ConvertibleTo<char>()));
  (void)absl::Seconds(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Seconds(static_cast<int64_t>(ConvertibleTo<char>()));
  (void)absl::Minutes(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Minutes(static_cast<int64_t>(ConvertibleTo<char>()));
  (void)absl::Hours(ConvertibleTo<char>());
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Hours(static_cast<int64_t>(ConvertibleTo<char>()));

  // User defined conversion to floating point type, followed by built-in conversion:
  (void)absl::Nanoseconds(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(ConvertibleTo<float>()));
  (void)absl::Microseconds(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Microseconds(static_cast<int64_t>(ConvertibleTo<float>()));
  (void)absl::Milliseconds(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Milliseconds(static_cast<int64_t>(ConvertibleTo<float>()));
  (void)absl::Seconds(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Seconds(static_cast<int64_t>(ConvertibleTo<float>()));
  (void)absl::Minutes(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Minutes(static_cast<int64_t>(ConvertibleTo<float>()));
  (void)absl::Hours(ConvertibleTo<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Hours(static_cast<int64_t>(ConvertibleTo<float>()));
}

void factoryFunctionNegative() {
  (void)absl::Nanoseconds(char{1});
  (void)absl::Nanoseconds(1);
  (void)absl::Nanoseconds(int64_t{1});
  (void)absl::Nanoseconds(1.0f);
  (void)absl::Microseconds(char{1});
  (void)absl::Microseconds(1);
  (void)absl::Microseconds(int64_t{1});
  (void)absl::Microseconds(1.0f);
  (void)absl::Milliseconds(char{1});
  (void)absl::Milliseconds(1);
  (void)absl::Milliseconds(int64_t{1});
  (void)absl::Milliseconds(1.0f);
  (void)absl::Seconds(char{1});
  (void)absl::Seconds(1);
  (void)absl::Seconds(int64_t{1});
  (void)absl::Seconds(1.0f);
  (void)absl::Minutes(char{1});
  (void)absl::Minutes(1);
  (void)absl::Minutes(int64_t{1});
  (void)absl::Minutes(1.0f);
  (void)absl::Hours(char{1});
  (void)absl::Hours(1);
  (void)absl::Hours(int64_t{1});
  (void)absl::Hours(1.0f);

  (void)absl::Nanoseconds(static_cast<int>(ConvertibleTo<int>()));
  (void)absl::Microseconds(static_cast<int>(ConvertibleTo<int>()));
  (void)absl::Milliseconds(static_cast<int>(ConvertibleTo<int>()));
  (void)absl::Seconds(static_cast<int>(ConvertibleTo<int>()));
  (void)absl::Minutes(static_cast<int>(ConvertibleTo<int>()));
  (void)absl::Hours(static_cast<int>(ConvertibleTo<int>()));
}

template <typename T> void templateForFactorySpecialization(T) {}
template <> void templateForFactorySpecialization<ConvertibleTo<int>>(ConvertibleTo<int> c) {
  (void)absl::Nanoseconds(c);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(c));
}

template <int N> void factoryNonTypeTemplateParamSpecialization() {
  (void)absl::Nanoseconds(N);
}

template <> void factoryNonTypeTemplateParamSpecialization<5>() {
  (void)absl::Nanoseconds(ConvertibleTo<int>());
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(ConvertibleTo<int>()));
}

template <typename T> void templateFactoryFix() {
  (void)absl::Nanoseconds(ConvertibleTo<int>());
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(ConvertibleTo<int>()));
}

template <typename T> void templateFactoryWarnOnly(T t) {
  (void)absl::Nanoseconds(t);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

template <typename T> void templateFactoryInstantiationBeforeDefinition(T t);

template <typename T> struct TemplateTypeFactoryWarnOnly {
  void memberA(T t) {
    (void)absl::Nanoseconds(t);
    // CHECK-MESSAGES: [[@LINE-1]]:29: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  }
  template <typename U> void memberB(U u) {
    (void)absl::Nanoseconds(u);
    // CHECK-MESSAGES: [[@LINE-1]]:29: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  }
};

void factoryInTemplates() {
  templateForFactorySpecialization(5);
  templateForFactorySpecialization(ConvertibleTo<int>());
  factoryNonTypeTemplateParamSpecialization<1>();
  factoryNonTypeTemplateParamSpecialization<5>();
  templateFactoryFix<int>();
  templateFactoryWarnOnly(ConvertibleTo<int>());
  templateFactoryInstantiationBeforeDefinition(ConvertibleTo<int>());
  TemplateTypeFactoryWarnOnly<ConvertibleTo<int>> t;
  t.memberA(ConvertibleTo<int>());
  t.memberB(ConvertibleTo<int>());
}

template <typename T> void templateFactoryInstantiationBeforeDefinition(T t) {
  (void)absl::Nanoseconds(t);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

#define ONLY_WARN_INSIDE_MACRO_FACTORY                                         \
  (void)absl::Nanoseconds(ConvertibleTo<int>())
#define T_CALL_FACTORTY_INSIDE_MACRO (void)absl::Nanoseconds(T())

template <typename T> void factoryTemplateAndMacro() {
  (void)absl::Nanoseconds(T_OBJECT);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  (void)absl::Nanoseconds(CONVERTIBLE_TMP);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(CONVERTIBLE_TMP))
  T_CALL_FACTORTY_INSIDE_MACRO;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
}

#define TEMPLATE_FACTORY_MACRO(factory)                                        \
  template <typename T> void TemplateFactoryInMacro(T t) { (void)factory(t); }

TEMPLATE_FACTORY_MACRO(absl::Nanoseconds)
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead

void factoryInMacros() {
  (void)absl::Nanoseconds(FUNCTION_MACRO(ConvertibleTo<int>()));
  // CHECK-MESSAGES: [[@LINE-1]]:42: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(FUNCTION_MACRO(ConvertibleTo<int>())));
  (void)absl::Nanoseconds(CONVERTIBLE_TMP);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  // CHECK-FIXES: (void)absl::Nanoseconds(static_cast<int64_t>(CONVERTIBLE_TMP))
  ONLY_WARN_INSIDE_MACRO_FACTORY;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: implicit conversion to 'int64_t' is deprecated in this context; use an explicit cast instead
  factoryTemplateAndMacro<ConvertibleTo<int>>();
  TemplateFactoryInMacro(ConvertibleTo<int>());
}

// This is a reduced test-case for PR39949 and manifested in this check.
namespace std {
template <typename _Tp>
_Tp declval();

template <typename _Functor, typename... _ArgTypes>
struct __res {
  template <typename... _Args>
  static decltype(declval<_Functor>()(_Args()...)) _S_test(int);

  template <typename...>
  static void _S_test(...);

  typedef decltype(_S_test<_ArgTypes...>(0)) type;
};

template <typename>
struct function;

template <typename... _ArgTypes>
struct function<void(_ArgTypes...)> {
  template <typename _Functor,
            typename = typename __res<_Functor, _ArgTypes...>::type>
  function(_Functor) {}
};
} // namespace std

typedef std::function<void(void)> F;

F foo() {
  return F([] {});
}
