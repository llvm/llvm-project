// RUN: %clang_cc1 -std=c89 -verify %s
// expected-no-diagnostics

// From: https://github.com/llvm/llvm-project/issues/15522#issue-1071059939
int logf = 5;
int main() {
return logf;
}
