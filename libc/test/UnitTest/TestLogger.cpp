#include "test/UnitTest/TestLogger.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h" // write_to_stderr
#include "src/__support/UInt128.h"

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

// bool specialization
template <> TestLogger &TestLogger::operator<<(bool cond) {
  return *this << (cond ? "true" : "false");
}

// void * specialization
template <> TestLogger &TestLogger::operator<<(void *addr) {
  return *this << "0x" << cpp::to_string(reinterpret_cast<uintptr_t>(addr));
}

template <typename T> TestLogger &TestLogger::operator<<(T t) {
  if constexpr (cpp::is_integral_v<T> && cpp::is_unsigned_v<T> &&
                sizeof(T) > sizeof(uint64_t)) {
    static_assert(sizeof(T) % 8 == 0, "Unsupported size of UInt");
    char buf[IntegerToString::hex_bufsize<T>()];
    IntegerToString::hex(t, buf, false);
    return *this << "0x" << cpp::string_view(buf, sizeof(buf));
  } else {
    return *this << cpp::to_string(t);
  }
}

// is_integral specializations
// char is already specialized to handle character
template TestLogger &TestLogger::operator<< <short>(short);
template TestLogger &TestLogger::operator<< <int>(int);
template TestLogger &TestLogger::operator<< <long>(long);
template TestLogger &TestLogger::operator<< <long long>(long long);
template TestLogger &TestLogger::operator<< <unsigned char>(unsigned char);
template TestLogger &TestLogger::operator<< <unsigned short>(unsigned short);
template TestLogger &TestLogger::operator<< <unsigned int>(unsigned int);
template TestLogger &TestLogger::operator<< <unsigned long>(unsigned long);
template TestLogger &
TestLogger::operator<< <unsigned long long>(unsigned long long);

#ifdef __SIZEOF_INT128__
template TestLogger &TestLogger::operator<< <__uint128_t>(__uint128_t);
#endif
template TestLogger &TestLogger::operator<< <cpp::UInt<128>>(cpp::UInt<128>);
template TestLogger &TestLogger::operator<< <cpp::UInt<192>>(cpp::UInt<192>);
template TestLogger &TestLogger::operator<< <cpp::UInt<256>>(cpp::UInt<256>);
template TestLogger &TestLogger::operator<< <cpp::UInt<320>>(cpp::UInt<320>);

// TODO: Add floating point formatting once it's supported by StringStream.

TestLogger tlog;

} // namespace testing
} // namespace __llvm_libc
