// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK

// RUN: %clangxx -x c++ -fsanitize=implicit-integer-sign-change -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-integer-sign-change -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-integer-sign-change -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-integer-sign-change -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK

void test_unsigned() {
  unsigned char x;

  x = 0;
  x++;
  x = 0;
  ++x;

  x = 0;
  x--;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:4: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'unsigned char' changed the value to 255 (8-bit, unsigned)
  x = 0;
  --x;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'unsigned char' changed the value to 255 (8-bit, unsigned)

  x = 1;
  x++;
  x = 1;
  ++x;

  x = 1;
  x--;
  x = 1;
  --x;

  x = 254;
  x++;
  x = 254;
  ++x;

  x = 254;
  x--;
  x = 254;
  --x;

  x = 255;
  x++;
  x = 255;
  ++x;

  x = 255;
  x--;
  x = 255;
  --x;
}

void test_signed() {
  signed char x;

  x = -128;
  x++;
  x = -128;
  ++x;

  x = -128;
  x--;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:4: runtime error: implicit conversion from type 'int' of value -129 (32-bit, signed) to type 'signed char' changed the value to 127 (8-bit, signed)
  x = -128;
  --x;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -129 (32-bit, signed) to type 'signed char' changed the value to 127 (8-bit, signed)

  x = -1;
  x++;
  x = -1;
  ++x;

  x = -1;
  x--;
  x = -1;
  --x;

  x = 0;
  x++;
  x = 0;
  ++x;

  x = 0;
  x--;
  x = 0;
  --x;

  x = 1;
  x++;
  x = 1;
  ++x;

  x = 1;
  x--;
  x = 1;
  --x;

  x = 127;
  x++;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:4: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'signed char' changed the value to -128 (8-bit, signed)
  x = 127;
  ++x;
  // CHECK: {{.*}}integer-sign-change-incdec.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'signed char' changed the value to -128 (8-bit, signed)

  x = 127;
  x--;
  x = 127;
  --x;
}

int main() {
  test_unsigned();
  test_signed();

  return 0;
}
