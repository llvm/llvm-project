// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=NON-OWNING %s \
// RUN:   performance-expensive-value-or %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::custom::CamelOptional;::custom::PascalOptional" \
// RUN:   }}'
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=OWNING %s \
// RUN:   performance-expensive-value-or %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::custom::CamelOptional;::custom::PascalOptional", \
// RUN:     performance-expensive-value-or.WarnOnOwnershipTaking: true \
// RUN:   }}'

#include <optional>
#include <string>

void consumeRef(const std::string &);
void consume(std::string s);

// Reference-friendly contexts: warn in both modes.

void positiveConstRefBinding(std::optional<std::string> opt,
                             const std::string &fallback) {
  const std::string &ref = opt.value_or(fallback);
  // CHECK-MESSAGES-NON-OWNING: :[[@LINE-1]]:32: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-OWNING: :[[@LINE-2]]:32: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-NON-OWNING: const std::string &ref = (opt ? *opt : fallback);
  // CHECK-FIXES-OWNING: const std::string &ref = (opt ? *opt : fallback);
}

void positiveConstRefParam(std::optional<std::string> opt,
                           const std::string &fallback) {
  consumeRef(opt.value_or(fallback));
  // CHECK-MESSAGES-NON-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-OWNING: :[[@LINE-2]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-NON-OWNING: consumeRef((opt ? *opt : fallback));
  // CHECK-FIXES-OWNING: consumeRef((opt ? *opt : fallback));
}

void positiveConstMemberCall(std::optional<std::string> opt,
                             const std::string &fallback) {
  auto len = opt.value_or(fallback).size();
  // CHECK-MESSAGES-NON-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-OWNING: :[[@LINE-2]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-FIXES-NON-OWNING: auto len = (opt ? *opt : fallback).size();
  // CHECK-FIXES-OWNING: auto len = (opt ? *opt : fallback).size();
}

// Ownership-taking contexts: only warn in OWNING mode.

void positiveOwnershipValue(std::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

struct LargeStruct {
  char data[128];
};

void positiveOwnershipLarge(std::optional<LargeStruct> opt) {
  auto val = opt.value_or(LargeStruct{});
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'LargeStruct'
}

struct SmallNonTrivial {
  int x;
  SmallNonTrivial(int x) : x(x) {}
  SmallNonTrivial(const SmallNonTrivial &o) : x(o.x) {}
};

void positiveOwnershipSmallNonTrivial(std::optional<SmallNonTrivial> opt) {
  auto val = opt.value_or(SmallNonTrivial{42});
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'SmallNonTrivial'
}

void positiveOwnershipByValueParam(std::optional<std::string> opt) {
  consume(opt.value_or("fallback"));
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:15: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

// Side-effect note tests.

std::string makeFallback();
void positiveSideEffectFallback(std::optional<std::string> opt) {
  auto val = opt.value_or(makeFallback());
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
  // CHECK-MESSAGES-OWNING: :[[@LINE-2]]:27: note: the fallback is always evaluated
}

// Constructing a temporary with a non-trivial destructor is not treated as a
// definite side effect, so no note fires here.
void positiveSideEffectTemporary(std::optional<std::string> opt) {
  auto val = opt.value_or(std::string("fallback"));
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}

// Template instantiation.

template <typename T> void positiveTemplate(std::optional<T> opt) {
  auto val = opt.value_or(T{});
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'
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
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:28: warning: 'value_or' copies expensive type 'LargeStruct'
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

// Alternative spellings and custom optional types.

namespace absl {
template <typename T> class optional {
public:
  T value_or(T default_value) const;
};
} // namespace absl

void positiveAbslDefault(absl::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
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
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'valueOr' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
}

void positiveValueOrPascal(custom::PascalOptional<std::string> opt) {
  auto val = opt.ValueOr("default");
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:18: warning: 'ValueOr' copies expensive type 'std::basic_string<char>'; consider avoiding the copy
}

// Macro test: verify the warning points to a useful location.

#define GET_OR_DEFAULT(opt, def) opt.value_or(def)

void positiveMacro(std::optional<std::string> opt) {
  auto val = GET_OR_DEFAULT(opt, "default");
  // CHECK-MESSAGES-OWNING: :[[@LINE-1]]:14: warning: 'value_or' copies expensive type 'std::basic_string<char>'
}
