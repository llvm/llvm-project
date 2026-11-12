// RUN: %check_clang_tidy %s bugprone-string-constructor %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-string-constructor.WarnOnLargeLength: true, \
// RUN:     bugprone-string-constructor.LargeLengthThreshold: 10, \
// RUN:     bugprone-string-constructor.StringNames: '::std::basic_string;::std::basic_string_view;::custom::String' \
// RUN:   }}"
#include <string>

namespace custom {
struct String {
  String(const char *, unsigned int size);
  String(const char *);
};
} // namespace custom

extern const char *kPtr;

void TestLargeLengthThreshold() {
  std::string s1(kPtr, 11);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter [bugprone-string-constructor]

  std::string s2(kPtr, 9);

  std::string_view sv1(kPtr, 11);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: suspicious large length parameter [bugprone-string-constructor]

  std::string s3(kPtr, 0x1000000);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter [bugprone-string-constructor]
}

void TestWarnOnLargeLengthAndThreshold() {
  std::string s1(kPtr, 0x1000000);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter [bugprone-string-constructor]

  std::string s2(20, 'x');
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter [bugprone-string-constructor]
}

void TestStringNames() {
  custom::String cs1(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: constructing string from nullptr is undefined behaviour [bugprone-string-constructor]
}
