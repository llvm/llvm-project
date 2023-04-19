#include "test/UnitTest/TestLogger.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h" // write_to_stderr

#include <stdint.h>

namespace __llvm_libc {
namespace testing {

// cpp::string_view specialization
template <>
TestLogger &TestLogger::operator<< <cpp::string_view>(cpp::string_view str) {
  __llvm_libc::write_to_stderr(str);
  return *this;
}

// cpp::string specialization
template <> TestLogger &TestLogger::operator<< <cpp::string>(cpp::string str) {
  return *this << static_cast<cpp::string_view>(str);
}

// const char* specialization
template <> TestLogger &TestLogger::operator<< <const char *>(const char *str) {
  return *this << cpp::string_view(str);
}

// char* specialization
template <> TestLogger &TestLogger::operator<< <char *>(char *str) {
  return *this << cpp::string_view(str);
}

// char specialization
template <> TestLogger &TestLogger::operator<<(char ch) {
  return *this << cpp::string_view(&ch, 1);
}

// void * specialization
template <> TestLogger &TestLogger::operator<<(void *addr) {
  return *this << "0x" << cpp::to_string(reinterpret_cast<uintptr_t>(addr));
}

template <typename T> TestLogger &TestLogger::operator<<(T t) {
  return *this << cpp::to_string(t);
}

// is_integral specializations
template TestLogger &TestLogger::operator<< <int>(int);
template TestLogger &TestLogger::operator<< <unsigned int>(unsigned int);
template TestLogger &TestLogger::operator<< <long>(long);
template TestLogger &TestLogger::operator<< <unsigned long>(unsigned long);
template TestLogger &TestLogger::operator<< <long long>(long long);
template TestLogger &
TestLogger::operator<< <unsigned long long>(unsigned long long);

// TODO: Add floating point formatting once it's supported by StringStream.

TestLogger tlog;

} // namespace testing
} // namespace __llvm_libc
