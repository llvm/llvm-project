// RUN: %check_clang_tidy %s readability-redundant-string-cstr %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-string-cstr.StringParameterFunctions: \
// RUN:              '::fmt::format; ::fmt::print; ::BaseLogger::operator(); ::BaseLogger::Log'} \
// RUN:             }" \
// RUN:   -- -isystem %clang_tidy_headers
#include <string>

namespace fmt {
  inline namespace v8 {
    template<typename ...Args>
    void print(const char *, Args &&...);
    template<typename ...Args>
    std::string format(const char *, Args &&...);
  }
}

namespace notfmt {
  inline namespace v8 {
    template<typename ...Args>
    void print(const char *, Args &&...);
    template<typename ...Args>
    std::string format(const char *, Args &&...);
  }
}

void fmt_print(const std::string &s1, const std::string &s2, const std::string &s3) {
  fmt::print("One:{}\n", s1.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: fmt::print("One:{}\n", s1);

  fmt::print("One:{} Two:{} Three:{}\n", s1.c_str(), s2, s3.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:58: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: fmt::print("One:{} Two:{} Three:{}\n", s1, s2, s3);
}

// There's no c_str() call here, so it shouldn't be touched
void fmt_print_no_cstr(const std::string &s1, const std::string &s2) {
    fmt::print("One: {}, Two: {}\n", s1, s2);
}

// This isn't fmt::print, so it shouldn't be fixed.
void not_fmt_print(const std::string &s1) {
    notfmt::print("One: {}\n", s1.c_str());
}

void fmt_format(const std::string &s1, const std::string &s2, const std::string &s3) {
  auto r1 = fmt::format("One:{}\n", s1.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: auto r1 = fmt::format("One:{}\n", s1);

  auto r2 = fmt::format("One:{} Two:{} Three:{}\n", s1.c_str(), s2, s3.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:53: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:69: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: auto r2 = fmt::format("One:{} Two:{} Three:{}\n", s1, s2, s3);
}

// There's are c_str() calls here, so it shouldn't be touched
void fmt_format_no_cstr(const std::string &s1, const std::string &s2) {
    fmt::format("One: {}, Two: {}\n", s1, s2);
}

// This is not fmt::format, so it shouldn't be fixed
std::string not_fmt_format(const std::string &s1) {
    return notfmt::format("One: {}\n", s1.c_str());
}

class BaseLogger {
public:
  template <typename... Args>
  void operator()(const char *fmt, Args &&...args) {
  }

  template <typename... Args>
  void Log(const char *fmt, Args &&...args) {
  }
};

class DerivedLogger : public BaseLogger {};
class DoubleDerivedLogger : public DerivedLogger {};
typedef DerivedLogger TypedefDerivedLogger;

void logger1(const std::string &s1, const std::string &s2, const std::string &s3) {
  BaseLogger LOGGER;

  LOGGER("%s\n", s1.c_str(), s2, s3.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:34: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGER("%s\n", s1, s2, s3);

  DerivedLogger LOGGER2;
  LOGGER2("%d %s\n", 42, s1.c_str(), s2.c_str(), s3);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:38: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGER2("%d %s\n", 42, s1, s2, s3);

  DoubleDerivedLogger LOGGERD;
  LOGGERD("%d %s\n", 42, s1.c_str(), s2, s3.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:42: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGERD("%d %s\n", 42, s1, s2, s3);

  TypedefDerivedLogger LOGGERT;
  LOGGERT("%d %s\n", 42, s1.c_str(), s2, s3.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:42: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGERT("%d %s\n", 42, s1, s2, s3);
}

void logger2(const std::string &s1, const std::string &s2) {
  BaseLogger LOGGER3;

  LOGGER3.Log("%s\n", s1.c_str(), s2.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:35: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGER3.Log("%s\n", s1, s2);

  DerivedLogger LOGGER4;
  LOGGER4.Log("%d %s\n", 42, s1.c_str(), s2.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES: :[[@LINE-2]]:42: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: LOGGER4.Log("%d %s\n", 42, s1, s2);
}

class NotLogger {
public:
  template <typename... Args>
  void operator()(const char *fmt, Args &&...args) {
  }

  template <typename... Args>
  void Log(const char *fmt, Args &&...args) {
  }
};

void Log(const char *fmt, ...);

void logger3(const std::string &s1)
{
  // Not BaseLogger or something derived from it
  NotLogger LOGGER;
  LOGGER("%s\n", s1.c_str());
  LOGGER.Log("%s\n", s1.c_str());

  // Free function not in StringParameterFunctions list
  Log("%s\n", s1.c_str());
}
