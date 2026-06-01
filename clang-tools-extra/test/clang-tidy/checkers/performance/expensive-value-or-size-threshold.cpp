// RUN: %check_clang_tidy -std=c++11-or-later %s performance-expensive-value-or %t \
// RUN:   -config='{CheckOptions: {performance-expensive-value-or.SizeThreshold: 8, performance-expensive-value-or.WarnOnOwnershipTaking: true}}'

#include <optional>

struct EightBytes {
  char d[8];
};

void negativeBoundary(std::optional<EightBytes> opt) {
  auto val = opt.value_or(EightBytes{});
}

struct NineBytes {
  char d[9];
};

void positiveBoundary(std::optional<NineBytes> opt) {
  auto val = opt.value_or(NineBytes{});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'NineBytes'
}
