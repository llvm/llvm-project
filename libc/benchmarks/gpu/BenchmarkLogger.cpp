#include "benchmarks/gpu/BenchmarkLogger.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"               // write_to_stderr
#include "src/__support/big_int.h"                 // is_big_int
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128
#include "src/__support/uint128.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace benchmarks {

// cpp::string_view specialization
template <>
BenchmarkLogger &
    BenchmarkLogger::operator<< <cpp::string_view>(cpp::string_view str) {
  LIBC_NAMESPACE::write_to_stderr(str);
  return *this;
}

// cpp::string specialization
template <>
BenchmarkLogger &BenchmarkLogger::operator<< <cpp::string>(cpp::string str) {
  return *this << static_cast<cpp::string_view>(str);
}

// const char* specialization
template <>
BenchmarkLogger &BenchmarkLogger::operator<< <const char *>(const char *str) {
  return *this << cpp::string_view(str);
}

// char* specialization
template <> BenchmarkLogger &BenchmarkLogger::operator<< <char *>(char *str) {
  return *this << cpp::string_view(str);
}

// char specialization
template <> BenchmarkLogger &BenchmarkLogger::operator<<(char ch) {
  return *this << cpp::string_view(&ch, 1);
}

// bool specialization
template <> BenchmarkLogger &BenchmarkLogger::operator<<(bool cond) {
  return *this << (cond ? "true" : "false");
}

// void * specialization
template <> BenchmarkLogger &BenchmarkLogger::operator<<(void *addr) {
  return *this << "0x" << cpp::to_string(reinterpret_cast<uintptr_t>(addr));
}

template <typename T> BenchmarkLogger &BenchmarkLogger::operator<<(T t) {
  if constexpr (is_big_int_v<T> ||
                (cpp::is_integral_v<T> && cpp::is_unsigned_v<T> &&
                 (sizeof(T) > sizeof(uint64_t)))) {
    static_assert(sizeof(T) % 8 == 0, "Unsupported size of UInt");
    const IntegerToString<T, radix::Hex::WithPrefix> buffer(t);
    return *this << buffer.view();
  } else {
    return *this << cpp::to_string(t);
  }
}

// is_integral specializations
// char is already specialized to handle character
template BenchmarkLogger &BenchmarkLogger::operator<< <short>(short);
template BenchmarkLogger &BenchmarkLogger::operator<< <int>(int);
template BenchmarkLogger &BenchmarkLogger::operator<< <long>(long);
template BenchmarkLogger &BenchmarkLogger::operator<< <long long>(long long);
template BenchmarkLogger &
    BenchmarkLogger::operator<< <unsigned char>(unsigned char);
template BenchmarkLogger &
    BenchmarkLogger::operator<< <unsigned short>(unsigned short);
template BenchmarkLogger &
    BenchmarkLogger::operator<< <unsigned int>(unsigned int);
template BenchmarkLogger &
    BenchmarkLogger::operator<< <unsigned long>(unsigned long);
template BenchmarkLogger &
    BenchmarkLogger::operator<< <unsigned long long>(unsigned long long);

#ifdef LIBC_TYPES_HAS_INT128
template BenchmarkLogger &
    BenchmarkLogger::operator<< <__uint128_t>(__uint128_t);
#endif // LIBC_TYPES_HAS_INT128
template BenchmarkLogger &BenchmarkLogger::operator<< <UInt<128>>(UInt<128>);
template BenchmarkLogger &BenchmarkLogger::operator<< <UInt<192>>(UInt<192>);
template BenchmarkLogger &BenchmarkLogger::operator<< <UInt<256>>(UInt<256>);
template BenchmarkLogger &BenchmarkLogger::operator<< <UInt<320>>(UInt<320>);

// TODO: Add floating point formatting once it's supported by StringStream.

BenchmarkLogger log;

} // namespace benchmarks
} // namespace LIBC_NAMESPACE_DECL
