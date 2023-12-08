// RUN: %check_clang_tidy %s abseil-duration-division %t

namespace absl {

class Duration {};

int operator/(Duration lhs, Duration rhs);

double FDivDuration(Duration num, Duration den);

}  // namespace absl

void TakesDouble(double);

#define MACRO_EQ(x, y) (x == y)
#define MACRO_DIVEQ(x,y,z) (x/y == z)
#define CHECK(x) (x)

void Positives() {
  absl::Duration d;

  const double num_double = d/d;
  // CHECK-MESSAGES: [[@LINE-1]]:30: warning: operator/ on absl::Duration objects performs integer division; did you mean to use FDivDuration()? [abseil-duration-division]
  // CHECK-FIXES: const double num_double = absl::FDivDuration(d, d);
  const float num_float = d/d;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: const float num_float = absl::FDivDuration(d, d);
  const auto SomeVal = 1.0 + d/d;
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: const auto SomeVal = 1.0 + absl::FDivDuration(d, d);
  if (MACRO_EQ(d/d, 0.0)) {}
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: if (MACRO_EQ(absl::FDivDuration(d, d), 0.0)) {}
  if (CHECK(MACRO_EQ(d/d, 0.0))) {}
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: if (CHECK(MACRO_EQ(absl::FDivDuration(d, d), 0.0))) {}

  // This one generates a message, but no fix.
  if (MACRO_DIVEQ(d, d, 0.0)) {}
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: if (MACRO_DIVEQ(d, d, 0.0)) {}
 
  TakesDouble(d/d);
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: operator/ on absl::Duration objects
  // CHECK-FIXES: TakesDouble(absl::FDivDuration(d, d));
}

void TakesInt(int);
template <class T>
void TakesGeneric(T);

void Negatives() {
  absl::Duration d;
  const int num_int = d/d;
  const long num_long = d/d;
  const short num_short = d/d;
  const char num_char = d/d;
  const auto num_auto = d/d;
  const auto SomeVal = 1 + d/d;

  TakesInt(d/d);
  TakesGeneric(d/d);
  // Explicit cast should disable the warning.
  const double num_cast1 = static_cast<double>(d/d);
  const double num_cast2 = (double)(d/d);
}

template <class T>
double DoubleDivision(T t1, T t2) {return t1/t2;}

//This also won't trigger a warning
void TemplateDivision() {
  absl::Duration d;
  DoubleDivision(d, d);
}
