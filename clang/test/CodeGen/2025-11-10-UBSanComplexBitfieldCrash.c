// Reduced from https://github.com/llvm/llvm-project/issues/166512
// RUN: %clang_cc1 %s -emit-obj -std=c23 -fsanitize=bool -o %t

struct {
  int : 7;
  int : 7;
  int : 7;
  bool bobf : 1;
} bits;
int main() { bits.bobf /= __builtin_complex(.125f, .125f); }
