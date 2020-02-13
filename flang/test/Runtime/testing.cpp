#include "testing.h"
#include "../../runtime/terminator.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static int failures{0};

// Override the Fortran runtime's Crash() for testing purposes
[[noreturn]] static void CatchCrash(const char *message, va_list &ap) {
  char buffer[1000];
  std::vsnprintf(buffer, sizeof buffer, message, ap);
  va_end(ap);
  throw std::string{buffer};
}

void StartTests() {
  Fortran::runtime::Terminator::RegisterCrashHandler(CatchCrash);
}

std::ostream &Fail() {
  ++failures;
  return std::cerr;
}

int EndTests() {
  if (failures == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL " << failures << " tests\n";
  }
  return failures != 0;
}

void SetCharacter(char *to, std::size_t n, const char *from) {
  auto len{std::strlen(from)};
  std::memcpy(to, from, std::min(len, n));
  if (len < n) {
    std::memset(to + len, ' ', n - len);
  }
}
