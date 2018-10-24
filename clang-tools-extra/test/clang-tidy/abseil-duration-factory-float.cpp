// RUN: %check_clang_tidy %s abseil-duration-factory-float %t

// Mimic the implementation of absl::Duration
namespace absl {

class Duration {};

Duration Nanoseconds(long long);
Duration Microseconds(long long);
Duration Milliseconds(long long);
Duration Seconds(long long);
Duration Minutes(long long);
Duration Hours(long long);

#define GENERATE_DURATION_FACTORY_OVERLOADS(NAME) \
  Duration NAME(float n);                         \
  Duration NAME(double n);                        \
  template <typename T>                           \
  Duration NAME(T n);

GENERATE_DURATION_FACTORY_OVERLOADS(Nanoseconds);
GENERATE_DURATION_FACTORY_OVERLOADS(Microseconds);
GENERATE_DURATION_FACTORY_OVERLOADS(Milliseconds);
GENERATE_DURATION_FACTORY_OVERLOADS(Seconds);
GENERATE_DURATION_FACTORY_OVERLOADS(Minutes);
GENERATE_DURATION_FACTORY_OVERLOADS(Hours);
#undef GENERATE_DURATION_FACTORY_OVERLOADS

}  // namespace absl

void ConvertFloatTest() {
  absl::Duration d;

  d = absl::Seconds(60.0);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(60);
  d = absl::Minutes(300.0);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Minutes [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Minutes(300);

  d = absl::Milliseconds(1e2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Milliseconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Milliseconds(100);
  d = absl::Seconds(3.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(3);
  d = absl::Seconds(3.);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(3);
  d = absl::Seconds(0x3.p0);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(3);
  d = absl::Seconds(0x3.p1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(6);


  // Ignored expressions
  d = absl::Seconds(.001);
  d = absl::Seconds(.100);
  d = ::absl::Seconds(1);
  d = ::absl::Minutes(1);
  d = ::absl::Hours(1);
  d = absl::Seconds(0x3.4p1);

  // Negative literals (we don't yet handle this case)
  d = absl::Seconds(-3.0);

  // This is bigger than we can safely fit in a positive int32, so we ignore it.
  d = absl::Seconds(1e12);

  int x;
  d = absl::Seconds(x);
  float y;
  d = absl::Minutes(y);

#define SECONDS(x) absl::Seconds(x)
  SECONDS(60);
#undef SECONDS
#define THIRTY 30.0
  d = absl::Seconds(THIRTY);
#undef THIRTY
}

template <int N>
void InTemplate() {
  absl::Duration d;

  d = absl::Seconds(N);  // 1
  // ^ No replacement here.

  d = absl::Minutes(1.0);  // 2
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Minutes [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Minutes(1);  // 2
}

void Instantiate() {
  InTemplate<60>();
  InTemplate<1>();
}

void ConvertCastTest() {
  absl::Duration d;

  d = absl::Seconds(static_cast<double>(5));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(5);

  d = absl::Minutes(static_cast<float>(5));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Minutes [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Minutes(5);

  d = absl::Seconds((double) 5);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(5);

  d = absl::Minutes((float) 5);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Minutes [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Minutes(5);

  d = absl::Seconds(double(5));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Seconds [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Seconds(5);

  d = absl::Minutes(float(5));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: use the integer version of absl::Minutes [abseil-duration-factory-float]
  // CHECK-FIXES: absl::Minutes(5);

  // This should not be flagged
  d = absl::Seconds(static_cast<int>(5.0));
  d = absl::Seconds((int) 5.0);
  d = absl::Seconds(int(5.0));
}
