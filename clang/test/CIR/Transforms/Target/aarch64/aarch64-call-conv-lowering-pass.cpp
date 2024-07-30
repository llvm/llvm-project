// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: @_Z4Voidv()
void Void(void) {
// CHECK:   cir.call @_Z4Voidv() : () -> ()
  Void();
}

// Test call conv lowering for trivial usinged integer cases.

// CHECK: @_Z4Boolb(%arg0: !cir.bool loc({{.+}})) -> !cir.bool
bool Bool(bool a) {
// CHECK:   cir.call @_Z4Boolb({{.+}}) : (!cir.bool) -> !cir.bool
  return Bool(a);
}

// CHECK: cir.func @_Z5UCharh(%arg0: !u8i loc({{.+}})) -> !u8i
unsigned char UChar(unsigned char c) {
  // CHECK: cir.call @_Z5UCharh(%2) : (!u8i) -> !u8i
  return UChar(c);
}
// CHECK: cir.func @_Z6UShortt(%arg0: !u16i loc({{.+}})) -> !u16i
unsigned short UShort(unsigned short s) {
  // CHECK: cir.call @_Z6UShortt(%2) : (!u16i) -> !u16i
  return UShort(s);
}
// CHECK: cir.func @_Z4UIntj(%arg0: !u32i loc({{.+}})) -> !u32i
unsigned int UInt(unsigned int i) {
  // CHECK: cir.call @_Z4UIntj(%2) : (!u32i) -> !u32i
  return UInt(i);
}
// CHECK: cir.func @_Z5ULongm(%arg0: !u64i loc({{.+}})) -> !u64i
unsigned long ULong(unsigned long l) {
  // CHECK: cir.call @_Z5ULongm(%2) : (!u64i) -> !u64i
  return ULong(l);
}
// CHECK: cir.func @_Z9ULongLongy(%arg0: !u64i loc({{.+}})) -> !u64i
unsigned long long ULongLong(unsigned long long l) {
  // CHECK: cir.call @_Z9ULongLongy(%2) : (!u64i) -> !u64i
  return ULongLong(l);
}


/// Test call conv lowering for trivial signed cases. ///

// CHECK: cir.func @_Z4Chara(%arg0: !s8i loc({{.+}})) -> !s8i
char Char(signed char c) {
  // CHECK: cir.call @_Z4Chara(%{{.+}}) : (!s8i) -> !s8i
  return Char(c);
}
// CHECK: cir.func @_Z5Shorts(%arg0: !s16i loc({{.+}})) -> !s16i
short Short(short s) {
  // CHECK: cir.call @_Z5Shorts(%{{.+}}) : (!s16i) -> !s16i
  return Short(s);
}
// CHECK: cir.func @_Z3Inti(%arg0: !s32i loc({{.+}})) -> !s32i
int Int(int i) {
  // CHECK: cir.call @_Z3Inti(%{{.+}}) : (!s32i) -> !s32i
  return Int(i);
}
// CHECK: cir.func @_Z4Longl(%arg0: !s64i loc({{.+}})) -> !s64i
long Long(long l) {
  // CHECK: cir.call @_Z4Longl(%{{.+}}) : (!s64i) -> !s64i
  return Long(l);
}
// CHECK: cir.func @_Z8LongLongx(%arg0: !s64i loc({{.+}})) -> !s64i
long long LongLong(long long l) {
  // CHECK: cir.call @_Z8LongLongx(%{{.+}}) : (!s64i) -> !s64i
  return LongLong(l);
}


/// Test call conv lowering for floating point. ///

// CHECK: cir.func @_Z5Floatf(%arg0: !cir.float loc({{.+}})) -> !cir.float
float Float(float f) {
  // cir.call @_Z5Floatf(%{{.+}}) : (!cir.float) -> !cir.float
  return Float(f);
}
// CHECK: cir.func @_Z6Doubled(%arg0: !cir.double loc({{.+}})) -> !cir.double
double Double(double d) {
  // cir.call @_Z6Doubled(%{{.+}}) : (!cir.double) -> !cir.double
  return Double(d);
}
