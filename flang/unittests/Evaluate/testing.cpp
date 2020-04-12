#include "testing.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace testing {

namespace {
int passes{0};
int failures{0};
} // namespace

static void BitBucket(const char *, ...) {}

static void PrintFailureDetails(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  fputs("\t", stderr);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
}

FailureDetailPrinter Test(
    const char *file, int line, const char *predicate, bool pass) {
  if (pass) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL: %s\n", file, line, predicate);
    return PrintFailureDetails;
  }
}

FailureDetailPrinter Match(const char *file, int line, std::uint64_t want,
    const char *gots, std::uint64_t got) {
  if (want == got) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL: %s == 0x%jx, not 0x%jx\n", file, line, gots,
        static_cast<std::uintmax_t>(got), static_cast<std::uintmax_t>(want));
    return PrintFailureDetails;
  }
}

FailureDetailPrinter Match(const char *file, int line, const char *want,
    const char *gots, const std::string &got) {
  if (want == got) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL: %s == \"%s\", not \"%s\"\n", file, line, gots,
        got.data(), want);
    return PrintFailureDetails;
  }
}

FailureDetailPrinter Match(const char *file, int line, const std::string &want,
    const char *gots, const std::string &got) {
  return Match(file, line, want.data(), gots, got);
}

FailureDetailPrinter Compare(const char *file, int line, const char *xs,
    const char *rel, const char *ys, std::uint64_t x, std::uint64_t y) {
  while (*rel == ' ') {
    ++rel;
  }
  bool pass{false};
  if (*rel == '<') {
    if (rel[1] == '=') {
      pass = x <= y;
    } else {
      pass = x < y;
    }
  } else if (*rel == '>') {
    if (rel[1] == '=') {
      pass = x >= y;
    } else {
      pass = x > y;
    }
  } else if (*rel == '=') {
    pass = x == y;
  } else if (*rel == '!') {
    pass = x != y;
  }
  if (pass) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL: %s[0x%jx] %s %s[0x%jx]\n", file, line, xs,
        static_cast<std::uintmax_t>(x), rel, ys,
        static_cast<std::uintmax_t>(y));
    return PrintFailureDetails;
  }
}

int Complete() {
  if (failures == 0) {
    if (passes == 1) {
      llvm::outs() << "single test PASSES\n";
    } else {
      llvm::outs() << "all " << passes << " tests PASS\n";
    }
    passes = 0;
    return EXIT_SUCCESS;
  } else {
    if (passes == 1) {
      llvm::errs() << "1 test passes, ";
    } else {
      llvm::errs() << passes << " tests pass, ";
    }
    if (failures == 1) {
      llvm::errs() << "1 test FAILS\n";
    } else {
      llvm::errs() << failures << " tests FAIL\n";
    }
    passes = failures = 0;
    return EXIT_FAILURE;
  }
}
} // namespace testing
