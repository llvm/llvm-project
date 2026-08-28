// RUN: %check_clang_tidy -std=c++20-or-later %s readability-redundant-parentheses %t

template <typename T>
constexpr bool valid() { return sizeof(T) >= 1; }

struct S { bool M; static constexpr bool SM = true; };
constexpr S Obj{true};
constexpr bool Arr[2] = {true, true};
constexpr bool Cond = true;

template <typename T>
requires (valid<T>())
constexpr T forward(T val) { return val; }

template <typename T>
void trailing(T) requires (valid<T>());

template <typename T> struct PartialSpec;
template <typename T>
requires (valid<T>()) struct PartialSpec<T *> {};

template <typename T>
concept Concept = (valid<T>());

template <typename T>
concept NestedRequirement = requires { requires (valid<T>()); };

auto Lambda = []<typename T> requires (valid<T>()) (T) {};

template <typename T>
requires (valid<T>()) && (valid<T>()) || (valid<T>())
void chained(T);

template <typename T>
requires (Obj.M) && (Arr[0])
void chainedLValue(T);

template <typename T>
requires ((valid<T *>()))
void nested(T);
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: requires (valid<T *>())

template <typename T>
concept SimpleRequirement = requires { (valid<T>()); };
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: concept SimpleRequirement = requires { valid<T>(); };

template <typename T>
requires (!(valid<T>()))
void negated(T);
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: requires (!valid<T>())

template <typename T>
requires (true)
void primaryLiteral(T);
// CHECK-MESSAGES: :[[@LINE-2]]:10: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: requires true

template <typename T>
requires (Cond) && (valid<T>())
void primaryChained(T);
// CHECK-MESSAGES: :[[@LINE-2]]:10: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: requires Cond && (valid<T>())

template <typename T>
requires (S::SM) void qualified(T);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES: requires S::SM void qualified(T);
