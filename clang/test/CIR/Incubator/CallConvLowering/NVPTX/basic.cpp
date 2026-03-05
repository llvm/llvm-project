// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir \
// RUN:            -fclangir-call-conv-lowering -emit-cir-flat -mmlir \
// RUN:            --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test call conv lowering for trivial cases. //

// CHECK: @_Z4Voidv()
void Void(void) {
  // CHECK:   cir.call @_Z4Voidv() : () -> ()
  Void();
}

// CHECK: @_Z4Boolb(%arg0: !cir.bool {cir.zeroext} loc({{.+}})) -> (!cir.bool {cir.zeroext})
bool Bool(bool a) {
  // CHECK:   cir.call @_Z4Boolb({{.+}}) : (!cir.bool) -> !cir.bool
  return Bool(a);
}

// CHECK: cir.func {{.*}} @_Z5UCharh(%arg0: !u8i {cir.zeroext} loc({{.+}})) -> (!u8i {cir.zeroext})
unsigned char UChar(unsigned char c) {
  // CHECK: cir.call @_Z5UCharh(%{{.+}}) : (!u8i) -> !u8i
  return UChar(c);
}

// CHECK: cir.func {{.*}} @_Z6UShortt(%arg0: !u16i {cir.zeroext} loc({{.+}})) -> (!u16i {cir.zeroext})
unsigned short UShort(unsigned short s) {
  // CHECK: cir.call @_Z6UShortt(%{{.+}}) : (!u16i) -> !u16i
  return UShort(s);
}

// CHECK: cir.func {{.*}} @_Z4UIntj(%arg0: !u32i loc({{.+}})) -> !u32i
unsigned int UInt(unsigned int i) {
  // CHECK: cir.call @_Z4UIntj(%{{.+}}) : (!u32i) -> !u32i
  return UInt(i);
}

// CHECK: cir.func {{.*}} @_Z5ULongm(%arg0: !u32i loc({{.+}})) -> !u32i
unsigned long ULong(unsigned long l) {
  // CHECK: cir.call @_Z5ULongm(%{{.+}}) : (!u32i) -> !u32i
  return ULong(l);
}

// CHECK: cir.func {{.*}} @_Z9ULongLongy(%arg0: !u64i loc({{.+}})) -> !u64i
unsigned long long ULongLong(unsigned long long l) {
  // CHECK: cir.call @_Z9ULongLongy(%{{.+}}) : (!u64i) -> !u64i
  return ULongLong(l);
}

// CHECK: cir.func {{.*}} @_Z4Chara(%arg0: !s8i {cir.signext} loc({{.+}})) -> (!s8i {cir.signext})
char Char(signed char c) {
  // CHECK: cir.call @_Z4Chara(%{{.+}}) : (!s8i) -> !s8i
  return Char(c);
}

// CHECK: cir.func {{.*}} @_Z5Shorts(%arg0: !s16i {cir.signext} loc({{.+}})) -> (!s16i {cir.signext})
short Short(short s) {
  // CHECK: cir.call @_Z5Shorts(%{{.+}}) : (!s16i) -> !s16i
  return Short(s);
}

// CHECK: cir.func {{.*}} @_Z3Inti(%arg0: !s32i loc({{.+}})) -> !s32i
int Int(int i) {
  // CHECK: cir.call @_Z3Inti(%{{.+}}) : (!s32i) -> !s32i
  return Int(i);
}

// CHECK: cir.func {{.*}} @_Z4Longl(%arg0: !s32i loc({{.+}})) -> !s32i
long Long(long l) {
  // CHECK: cir.call @_Z4Longl(%{{.+}}) : (!s32i) -> !s32i
  return Long(l);
}

// CHECK: cir.func {{.*}} @_Z8LongLongx(%arg0: !s64i loc({{.+}})) -> !s64i
long long LongLong(long long l) {
  // CHECK: cir.call @_Z8LongLongx(%{{.+}}) : (!s64i) -> !s64i
  return LongLong(l);
}


// Check for structs.

struct Struct {
  int a, b, c, d, e;
};

// CHECK: cir.func {{.*}} @_Z10StructFuncv() -> !rec_Struct
Struct StructFunc() {
  return { 0, 1, 2, 3, 4 };
}
