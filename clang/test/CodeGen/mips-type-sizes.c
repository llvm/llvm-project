// RUN: %clang_cc1 -triple mips-none-linux-gnu -emit-llvm -w -o - %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -target-abi n32 -o - %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -o - %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

int check_char(void) {
  return sizeof(char);
// ALL: ret i32 1
}

int check_short(void) {
  return sizeof(short);
// ALL: ret i32 2
}

int check_int(void) {
  return sizeof(int);
// ALL: ret i32 4
}

int check_long(void) {
  return sizeof(long);
// O32: ret i32 4
// N32: ret i32 4
// N64: ret i32 8
}

int check_longlong(void) {
  return sizeof(long long);
// ALL: ret i32 8
}

int check_fp16(void) {
  return sizeof(__fp16);
// ALL: ret i32 2
}

int check_float(void) {
  return sizeof(float);
// ALL: ret i32 4
}

int check_double(void) {
  return sizeof(double);
// ALL: ret i32 8
}

int check_longdouble(void) {
  return sizeof(long double);
// O32: ret i32 8
// N32: ret i32 16
// N64: ret i32 16
}

int check_floatComplex(void) {
  return sizeof(float _Complex);
// ALL: ret i32 8
}

int check_doubleComplex(void) {
  return sizeof(double _Complex);
// ALL: ret i32 16
}

int check_longdoubleComplex(void) {
  return sizeof(long double _Complex);
// O32: ret i32 16
// N32: ret i32 32
// N64: ret i32 32
}

int check_bool(void) {
  return sizeof(_Bool);
// ALL: ret i32 1
}

int check_wchar(void) {
  return sizeof(__WCHAR_TYPE__);
// ALL: ret i32 4
}

int check_wchar_is_unsigned(void) {
  return (__WCHAR_TYPE__)-1 > (__WCHAR_TYPE__)0;
// ALL: ret i32 0
}

int check_ptr(void) {
  return sizeof(void *);
// O32: ret i32 4
// N32: ret i32 4
// N64: ret i32 8
}

