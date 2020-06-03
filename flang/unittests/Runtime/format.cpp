// Tests basic FORMAT string traversal

#include "testing.h"
#include "../runtime/format-implementation.h"
#include "../runtime/io-error.h"
#include <cstdarg>
#include <cstring>
#include <string>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;
using namespace std::literals::string_literals;

using Results = std::vector<std::string>;

// A test harness context for testing FormatControl
class TestFormatContext : public IoErrorHandler {
public:
  using CharType = char;
  TestFormatContext() : IoErrorHandler{"format.cpp", 1} {}
  bool Emit(const char *, std::size_t);
  bool Emit(const char16_t *, std::size_t);
  bool Emit(const char32_t *, std::size_t);
  bool AdvanceRecord(int = 1);
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);
  void Report(const DataEdit &);
  void Check(Results &);
  Results results;
  MutableModes &mutableModes() { return mutableModes_; }

private:
  MutableModes mutableModes_;
};

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

void TestFormatContext::HandleAbsolutePosition(std::int64_t n) {
  results.push_back("T"s + std::to_string(n));
}

void TestFormatContext::HandleRelativePosition(std::int64_t n) {
  if (n < 0) {
    results.push_back("TL"s + std::to_string(-n));
  } else {
    results.push_back(std::to_string(n) + 'X');
  }
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
    Fail() << "expected:";
    for (const std::string &s : expect) {
      llvm::errs() << ' ' << s;
    }
    llvm::errs() << "\ngot:";
    for (const std::string &s : results) {
      llvm::errs() << ' ' << s;
    }
    llvm::errs() << '\n';
  }
  expect.clear();
  results.clear();
}

static void Test(int n, const char *format, Results &&expect, int repeat = 1) {
  TestFormatContext context;
  llvm::errs() << "In: " << format << '\n';
  FormatControl<TestFormatContext> control{
      context, format, std::strlen(format)};
  try {
    for (int j{0}; j < n; ++j) {
      context.Report(control.GetNextDataEdit(context, repeat));
    }
    control.Finish(context);
    if (int iostat{context.GetIoStat()}) {
      context.Crash("GetIoStat() == %d", iostat);
    }
  } catch (const std::string &crash) {
    context.results.push_back("Crash:"s + crash);
  }
  context.Check(expect);
}

int main() {
  StartTests();
  Test(1, "('PI=',F9.7)", Results{"'PI='", "F9.7"});
  Test(1, "(3HPI=F9.7)", Results{"'PI='", "F9.7"});
  Test(1, "(3HPI=/F9.7)", Results{"'PI='", "/", "F9.7"});
  Test(2, "('PI=',F9.7)", Results{"'PI='", "F9.7", "/", "'PI='", "F9.7"});
  Test(2, "(2('PI=',F9.7),'done')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7", "'done'"});
  Test(2, "(3('PI=',F9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(2, "(*('PI=',f9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(1, "(3F9.7)", Results{"2*F9.7"}, 2);
  // Test(1, "(10x)", Results{"10X"});
  // Test(1, "(10X)", Results{"10X"});
  // Test(2, "(10X,2Hhi)", Results{"10X", "'hi'"});
  //Test(2, "(10X,2hhi)", Results{"10X", "'hi'"});
	Test(0, "(\x0221\x022)", Results{""});
	Test(0, "(\x022 \x022,10x,\x022FORTRAN COMPILER VALIDATION SYSTEM\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,21x,\x022VERSION 1.0\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,10x,\x022FOR OFFICIAL USE ONLY - COPYRIGHT 1978\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,18x,\x022SUBSET LEVEL TEST\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,5x,\x022TEST\x022,5x,\x022PASS/FAIL\x022,5x,\x022COMPUTED\x022,8x,\x022CORRECT\x022)", Results{""});
	Test(0, "(\x022 \x022,5x,\x022----------------------------------------------\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4x,i5,7x,\x022FAIL\x022,10x,i6,9x,i6)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,5x,\x022----------------------------------------------\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,20x,\x022END OF PROGRAM FM004\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,15x,i5,\x022 ERRORS ENCOUNTERED\x022)", Results{""});
	Test(0, "(\x022 \x022,15x,i5,\x022 TESTS PASSED\x022)", Results{""});
	Test(0, "(\x022 \x022,15x,i5,\x022 TESTS DELETED\x022)", Results{""});
	Test(0, "(\x0221\x022)", Results{""});
	Test(0, "(\x022 \x022,10X,\x022FORTRAN COMPILER VALIDATION SYSTEM\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,21X,\x022VERSION 1.0\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,10X,\x022FOR OFFICIAL USE ONLY - COPYRIGHT 1978\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,18X,\x022SUBSET LEVEL TEST\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,5X,\x022TEST\x022,5X,\x022PASS/FAIL\x022,5X,\x022COMPUTED\x022,8X,\x022CORRECT\x022)", Results{""});
	Test(0, "(\x022 \x022,5X,\x022----------------------------------------------\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022DELETED\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022PASS\x022)", Results{""});
	Test(0, "(\x022 \x022,4X,i5,7X,\x022FAIL\x022,10X,i6,9X,i6)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,5X,\x022----------------------------------------------\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,20X,\x022END OF PROGRAM FM004\x022)", Results{""});
	Test(0, "(\x022 \x022)", Results{""});
	Test(0, "(\x022 \x022,15X,i5,\x022 ERRORS ENCOUNTERED\x022)", Results{""});
	Test(0, "(\x022 \x022,15X,i5,\x022 TESTS PASSED\x022)", Results{""});
	Test(0, "(\x022 \x022,15X,i5,\x022 TESTS DELETED\x022)", Results{""});
  return EndTests();
}
