// RUN: %check_clang_tidy -std=c++17-or-later -check-suffix=DEFAULT %s performance-expensive-value-or %t \
// RUN:   -config='{CheckOptions: {performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::custom::CamelOptional;::custom::PascalOptional"}}'
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffix=AGGRESSIVE %s performance-expensive-value-or %t \
// RUN:   -config='{CheckOptions: {performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::custom::CamelOptional;::custom::PascalOptional", performance-expensive-value-or.WarnOnOwnershipTaking: true}}'

#include <optional>
#include <string>
#include <utility>

void consumeRef(const std::string &);
void consume(std::string s);

// Reference-friendly contexts: warn in both default and aggressive modes.

void positiveConstRefBinding(std::optional<std::string> opt,
                             const std::string &fallback) {
  const std::string &ref = opt.value_or(fallback);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:32: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-2]]:32: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-DEFAULT: const std::string &ref = (opt ? *opt : fallback);
  // CHECK-FIXES-AGGRESSIVE: const std::string &ref = (opt ? *opt : fallback);
}

void positiveConstRefParam(std::optional<std::string> opt,
                           const std::string &fallback) {
  consumeRef(opt.value_or(fallback));
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-2]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-DEFAULT: consumeRef((opt ? *opt : fallback));
  // CHECK-FIXES-AGGRESSIVE: consumeRef((opt ? *opt : fallback));
}

void positiveConstMemberCall(std::optional<std::string> opt,
                             const std::string &fallback) {
  auto len = opt.value_or(fallback).size();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-2]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-DEFAULT: auto len = (opt ? *opt : fallback).size();
  // CHECK-FIXES-AGGRESSIVE: auto len = (opt ? *opt : fallback).size();
}

// Ownership-taking contexts: only warn in aggressive mode.

void positiveOwnershipValue(std::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

struct LargeStruct {
  char data[128];
};

void positiveOwnershipLarge(std::optional<LargeStruct> opt) {
  auto val = opt.value_or(LargeStruct{});
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'LargeStruct'
}

struct SmallNonTrivial {
  int x;
  SmallNonTrivial(int x) : x(x) {}
  SmallNonTrivial(const SmallNonTrivial &o) : x(o.x) {}
};

void positiveOwnershipSmallNonTrivial(std::optional<SmallNonTrivial> opt) {
  auto val = opt.value_or(SmallNonTrivial{42});
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'SmallNonTrivial'
}

void positiveOwnershipByValueParam(std::optional<std::string> opt) {
  consume(opt.value_or("fallback"));
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:15: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

// Side-effect note tests (aggressive mode).

std::string makeFallback();
void positiveSideEffectFallback(std::optional<std::string> opt) {
  auto val = opt.value_or(makeFallback());
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-2]]:27: note: the fallback is always evaluated
}

// Constructing a temporary with a non-trivial destructor is not treated as a
// definite side effect, so no note fires here.
void positiveSideEffectTemporary(std::optional<std::string> opt) {
  auto val = opt.value_or(std::string("fallback"));
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

// Template instantiation (aggressive mode).

template <typename T> void positiveTemplate(std::optional<T> opt) {
  auto val = opt.value_or(T{});
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}
void instantiate() {
  positiveTemplate(std::optional<std::string>{});
}

// Rvalue optional tests.

std::optional<std::string> getOpt();
void negativeRvalueOptional() {
  auto val = getOpt().value_or("default");
}

std::optional<LargeStruct> getLargeOpt();
void positiveRvalueNoMoveAdvantage() {
  auto val = getLargeOpt().value_or(LargeStruct{});
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:28: warning: 'value_or' copies expensive type 'LargeStruct'
}

// Negative cases: no warning in either mode.

void negativeInt(std::optional<int> opt) {
  auto val = opt.value_or(0);
}

struct SmallPOD {
  char x;
  char y;
};

void negativeSmallPOD(std::optional<SmallPOD> opt) {
  auto val = opt.value_or(SmallPOD{0, 0});
}

struct SixteenBytes {
  char d[16];
};

void negativeBoundary(std::optional<SixteenBytes> opt) {
  auto val = opt.value_or(SixteenBytes{});
}

// Non-trivially-copyable (due to user-defined copy assignment) but copy
// construction itself is trivial. No point warning about an inexpensive copy.
struct TrivialCopyNonTrivialAssign {
  int x;
  TrivialCopyNonTrivialAssign() = default;
  TrivialCopyNonTrivialAssign(const TrivialCopyNonTrivialAssign &) = default;
  TrivialCopyNonTrivialAssign &operator=(const TrivialCopyNonTrivialAssign &);
  ~TrivialCopyNonTrivialAssign() = default;
};

void negativeTrivialCopyNonTrivialAssign(
    std::optional<TrivialCopyNonTrivialAssign> opt) {
  auto val = opt.value_or(TrivialCopyNonTrivialAssign{});
}

// Alternative spellings and custom optional types (aggressive mode).

namespace absl {
template <typename T> class optional {
public:
  T value_or(T default_value) const;
};
} // namespace absl

void positiveAbslDefault(absl::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
}

namespace custom {
template <typename T> class CamelOptional {
public:
  T valueOr(T default_value) const;
};
template <typename T> class PascalOptional {
public:
  T ValueOr(T default_value) const;
};
} // namespace custom

void positiveValueOr(custom::CamelOptional<std::string> opt) {
  auto val = opt.valueOr("default");
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'valueOr' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
}

void positiveValueOrPascal(custom::PascalOptional<std::string> opt) {
  auto val = opt.ValueOr("default");
  // CHECK-MESSAGES-AGGRESSIVE: :[[@LINE-1]]:18: warning: 'ValueOr' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
}
