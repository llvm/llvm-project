/// Check that ABI is correctly implemented.
///
///   1. Check that all integer arguments and return values less than 64 bits
///      are sign/zero extended.
///   2. Check that all complex arguments and return values are placed in
///      registers if it is possible.  Not treat it as aggregate.
///   3. Check that a function declared without argument type declarations is
///      treated as VARARGS (in order to place arguments in both registers and
///      memory locations in the back end)

// RUN: %clang_cc1 -triple ve-linux-gnu -emit-llvm -Wno-strict-prototypes %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} signext i8 @fun_si8(i8 noundef signext %a, i8 noundef signext %b) #0 {
char fun_si8(char a, char b) {
  return a;
}

// CHECK-LABEL: define{{.*}} zeroext i8 @fun_zi8(i8 noundef zeroext %a, i8 noundef zeroext %b) #0 {
unsigned char fun_zi8(unsigned char a, unsigned char b) {
  return a;
}

// CHECK-LABEL: define{{.*}} signext i16 @fun_si16(i16 noundef signext %a, i16 noundef signext %b) #0 {
short fun_si16(short a, short b) {
  return a;
}

// CHECK-LABEL: define{{.*}} zeroext i16 @fun_zi16(i16 noundef zeroext %a, i16 noundef zeroext %b) #0 {
unsigned short fun_zi16(unsigned short a, unsigned short b) {
  return a;
}

// CHECK-LABEL: define{{.*}} signext i32 @fun_si32(i32 noundef signext %a, i32 noundef signext %b) #0 {
int fun_si32(int a, int b) {
  return a;
}

// CHECK-LABEL: define{{.*}} zeroext i32 @fun_zi32(i32 noundef zeroext %a, i32 noundef zeroext %b) #0 {
unsigned int fun_zi32(unsigned int a, unsigned int b) {
  return a;
}

// CHECK-LABEL: define{{.*}} i64 @fun_si64(i64 noundef %a, i64 noundef %b) #0 {
long fun_si64(long a, long b) {
  return a;
}

// CHECK-LABEL: define{{.*}} i64 @fun_zi64(i64 noundef %a, i64 noundef %b) #0 {
unsigned long fun_zi64(unsigned long a, unsigned long b) {
  return a;
}

// CHECK-LABEL: define{{.*}} i128 @fun_si128(i128 noundef %a, i128 noundef %b) #0 {
__int128 fun_si128(__int128 a, __int128 b) {
  return a;
}

// CHECK-LABEL: define{{.*}} i128 @fun_zi128(i128 noundef %a, i128 noundef %b) #0 {
unsigned __int128 fun_zi128(unsigned __int128 a, unsigned __int128 b) {
  return a;
}

// CHECK-LABEL: define{{.*}} float @fun_float(float noundef %a, float noundef %b) #0 {
float fun_float(float a, float b) {
  return a;
}

// CHECK-LABEL: define{{.*}} double @fun_double(double noundef %a, double noundef %b) #0 {
double fun_double(double a, double b) {
  return a;
}

// CHECK-LABEL: define{{.*}} fp128 @fun_quad(fp128 noundef %a, fp128 noundef %b) #0 {
long double fun_quad(long double a, long double b) {
  return a;
}

// CHECK-LABEL: define{{.*}} { float, float } @fun_fcomplex(float noundef %a.coerce0, float noundef %a.coerce1, float noundef %b.coerce0, float noundef %b.coerce1) #0 {
float __complex__ fun_fcomplex(float __complex__ a, float __complex__ b) {
  return a;
}

// CHECK-LABEL: define{{.*}} { double, double } @fun_dcomplex(double noundef %a.coerce0, double noundef %a.coerce1, double noundef %b.coerce0, double noundef %b.coerce1) #0 {
double __complex__ fun_dcomplex(double __complex__ a, double __complex__ b) {
  return a;
}

// CHECK-LABEL: define{{.*}} { fp128, fp128 } @fun_qcomplex(fp128 noundef %a.coerce0, fp128 noundef %a.coerce1, fp128 noundef %b.coerce0, fp128 noundef %b.coerce1) #0 {
long double __complex__ fun_qcomplex(long double __complex__ a, long double __complex__ b) {
  return a;
}

extern int hoge();
void func(void) {
  // CHECK: %call = call signext i32 (i32, i32, i32, i32, i32, i32, i32, ...) @hoge(i32 noundef signext 1, i32 noundef signext 2, i32 noundef signext 3, i32 noundef signext 4, i32 noundef signext 5, i32 noundef signext 6, i32 noundef signext 7)
  hoge(1, 2, 3, 4, 5, 6, 7);
}
