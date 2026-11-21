#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MODERNIZE_INPUTS_USE_BIT_CAST_HEADER_H
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MODERNIZE_INPUTS_USE_BIT_CAST_HEADER_H

// CHECK-FIXES: #include <bit>

inline void header_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning [modernize-use-bit-cast]
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MODERNIZE_INPUTS_USE_BIT_CAST_HEADER_H
