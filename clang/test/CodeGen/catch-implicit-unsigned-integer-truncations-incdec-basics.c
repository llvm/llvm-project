// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation -fsanitize-recover=implicit-unsigned-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-LABEL: @t0(
unsigned short t0(unsigned short x) {
#line 100
  x++;
  return x;
}
// CHECK-LABEL: @t1(
unsigned short t1(unsigned short x) {
#line 200
  x--;
  return x;
}
// CHECK-LABEL: @t2(
unsigned short t2(unsigned short x) {
#line 300
  ++x;
  return x;
}
// CHECK-LABEL: @t3(
unsigned short t3(unsigned short x) {
#line 400
  --x;
  return x;
}

// CHECK-LABEL: @t4(
signed short t4(signed short x) {
#line 500
  x++;
  return x;
}
// CHECK-LABEL: @t5(
signed short t5(signed short x) {
#line 600
  x--;
  return x;
}
// CHECK-LABEL: @t6(
signed short t6(signed short x) {
#line 700
  ++x;
  return x;
}
// CHECK-LABEL: @t7(
signed short t7(signed short x) {
#line 800
  --x;
  return x;
}

// CHECK-LABEL: @t8(
unsigned char t8(unsigned char x) {
#line 900
  x++;
  return x;
}
// CHECK-LABEL: @t9(
unsigned char t9(unsigned char x) {
#line 1000
  x--;
  return x;
}
// CHECK-LABEL: @t10(
unsigned char t10(unsigned char x) {
#line 1100
  ++x;
  return x;
}
// CHECK-LABEL: @t11(
unsigned char t11(unsigned char x) {
#line 1200
  --x;
  return x;
}

// CHECK-LABEL: @t12(
signed char t12(signed char x) {
#line 1300
  x++;
  return x;
}
// CHECK-LABEL: @t13(
signed char t13(signed char x) {
#line 1400
  x--;
  return x;
}
// CHECK-LABEL: @t14(
signed char t14(signed char x) {
#line 1500
  ++x;
  return x;
}
// CHECK-LABEL: @t15(
signed char t15(signed char x) {
#line 1600
  --x;
  return x;
}
