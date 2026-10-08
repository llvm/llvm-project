// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-bit-cast %t

void *memcpy(void *To, const void *From, int Size);

namespace std {
void *memcpy(void *To, const void *From, __SIZE_TYPE__ Size);
}

void nonstandard_size_parameter_case() {
  float src = 1.0f;
  unsigned int dst;
  ::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void standard_size_parameter_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}
