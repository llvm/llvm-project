// RUN: %clang_cc1  %s -emit-llvm -o - | not grep builtin
double sqrt(double x);
void zsqrtxxx(float num) {
   num = sqrt(num);
}
