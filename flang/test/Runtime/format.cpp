// Tests basic FORMAT string traversal

#include "../runtime/format-implementation.h"
#include "../runtime/terminator.h"
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;
using namespace std::literals::string_literals;

static int failures{0};
using Results = std::vector<std::string>;

// A test harness context for testing FormatControl
class TestFormatContext : public Terminator {
public:
  using CharType = char;
  TestFormatContext() : Terminator{"format.cpp", 1} {}
  bool Emit(const char *, std::size_t);
  bool Emit(const char16_t *, std::size_t);
  bool Emit(const char32_t *, std::size_t);
  bool AdvanceRecord(int = 1);
  bool HandleRelativePosition(std::int64_t);
  bool HandleAbsolutePosition(std::int64_t);
  void Report(const DataEdit &);
  void Check(Results &);
  Results results;
  MutableModes &mutableModes() { return mutableModes_; }

private:
  MutableModes mutableModes_;
};

// Override the runtime's Crash() for testing purposes
[[noreturn]] void Fortran::runtime::Terminator::Crash(
    const char *message, ...) const {
  std::va_list ap;
  va_start(ap, message);
  char buffer[1000];
  std::vsnprintf(buffer, sizeof buffer, message, ap);
  va_end(ap);
  throw std::string{buffer};
}

bool TestFormatContext::Emit(const char *s, std::size_t len) {
  std::string str{s, len};
  results.push_back("'"s + str + '\'');
  return true;
}
bool TestFormatContext::Emit(const char16_t *, std::size_t) {
  Crash("TestFormatContext::Emit(const char16_t *) called");
  return false;
}
bool TestFormatContext::Emit(const char32_t *, std::size_t) {
  Crash("TestFormatContext::Emit(const char32_t *) called");
  return false;
}

bool TestFormatContext::AdvanceRecord(int n) {
  while (n-- > 0) {
    results.emplace_back("/");
  }
  return true;
}

bool TestFormatContext::HandleAbsolutePosition(std::int64_t n) {
  results.push_back("T"s + std::to_string(n));
  return true;
}

bool TestFormatContext::HandleRelativePosition(std::int64_t n) {
  if (n < 0) {
    results.push_back("TL"s + std::to_string(-n));
  } else {
    results.push_back(std::to_string(n) + 'X');
  }
  return true;
}

void TestFormatContext::Report(const DataEdit &edit) {
  std::string str{edit.descriptor};
  if (edit.repeat != 1) {
    str = std::to_string(edit.repeat) + '*' + str;
  }
  if (edit.variation) {
    str += edit.variation;
  }
  if (edit.width) {
    str += std::to_string(*edit.width);
  }
  if (edit.digits) {
    str += "."s + std::to_string(*edit.digits);
  }
  if (edit.expoDigits) {
    str += "E"s + std::to_string(*edit.expoDigits);
  }
  // modes?
  results.push_back(str);
}

void TestFormatContext::Check(Results &expect) {
  if (expect != results) {
    std::cerr << "expected:";
    for (const std::string &s : expect) {
      std::cerr << ' ' << s;
    }
    std::cerr << "\ngot:";
    for (const std::string &s : results) {
      std::cerr << ' ' << s;
    }
    std::cerr << '\n';
    ++failures;
  }
  expect.clear();
  results.clear();
}

static void Test(int n, const char *format, Results &&expect, int repeat = 1) {
  TestFormatContext context;
  FormatControl<TestFormatContext> control{
      context, format, std::strlen(format)};
  try {
    for (int j{0}; j < n; ++j) {
      context.Report(control.GetNextDataEdit(context, repeat));
    }
    control.FinishOutput(context);
  } catch (const std::string &crash) {
    context.results.push_back("Crash:"s + crash);
  }
  context.Check(expect);
}

int main() {
  Test(1, "('PI=',F9.7)", Results{"'PI='", "F9.7"});
  Test(1, "(3HPI=F9.7)", Results{"'PI='", "F9.7"});
  Test(1, "(3HPI=/F9.7)", Results{"'PI='", "/", "F9.7"});
  Test(2, "('PI=',F9.7)", Results{"'PI='", "F9.7", "/", "'PI='", "F9.7"});
  Test(2, "(2('PI=',F9.7),'done')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7", "'done'"});
  Test(2, "(3('PI=',F9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(2, "(*('PI=',F9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(1, "(3F9.7)", Results{"2*F9.7"}, 2);
  return failures > 0;
}
