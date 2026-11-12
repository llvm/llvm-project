// RUN: %clang_analyze_cc1 -analyzer-checker=core,apiModeling -verify %s

// expected-no-diagnostics

namespace llvm {
template <typename>
void cast(...);
void a() { cast<int>(int()); } // no-crash
} // namespace llvm
