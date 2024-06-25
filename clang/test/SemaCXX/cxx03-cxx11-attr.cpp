// RUN: %clang_cc1 -std=c++03 -fsyntax-only %s

// Ensure that __has_cpp_attribute and argument parsing work in C++03

#if !__has_cpp_attribute(nodiscard)
#  error
#endif

[[gnu::assume_aligned(4)]] void* g() { return __nullptr; }
