// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t

float ceil(float);
namespace std {
double ceil(double);
long double floor(long double);
} // namespace std

namespace floats {

struct ConvertsToFloat {
  operator float() const { return 0.5; }
};

float operator "" _Pa(unsigned long long);

void not_ok(double d) {
  int i = 0;
  i = d;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = static_cast<float>(d);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = ConvertsToFloat();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = 15_Pa;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
}

void not_ok_binary_ops(double d) {
  int i = 0;
  i += 0.5;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += d;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  // We warn on the following even though it's not dangerous because there is no
  // reason to use a double literal here.
  // TODO(courbet): Provide an automatic fix.
  i += 2.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += 2.0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]

  i *= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i /= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += (double)0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
}

void ok(double d) {
  int i = 0;
  i = 1;
  i = static_cast<int>(0.5);
  i = static_cast<int>(d);
  i = std::ceil(0.5);
  i = ::std::floor(0.5);
  {
    using std::ceil;
    i = ceil(0.5f);
  }
  i = ceil(0.5f);
}

void ok_binary_ops(double d) {
  int i = 0;
  i += 1;
  i += static_cast<int>(0.5);
  i += static_cast<int>(d);
  i += (int)d;
  i += std::ceil(0.5);
  i += ::std::floor(0.5);
  {
    using std::ceil;
    i += ceil(0.5f);
  }
  i += ceil(0.5f);
}

// We're bailing out in templates and macros.
template <typename T1, typename T2>
void f(T1 one, T2 two) {
  one += two;
}

void template_context() {
  f(1, 2);
  f(1, .5);
}

#define DERP(i, j) (i += j)

void macro_context() {
  int i = 0;
  DERP(i, 2);
  DERP(i, .5);
}

}  // namespace floats
