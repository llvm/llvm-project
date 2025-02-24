// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test call conv lowering for trivial cases. //

// CHECK: @_Z4Voidv()
void Void(void) {
// CHECK:   cir.call @_Z4Voidv() : () -> ()
  Void();
}

// Test call conv lowering for trivial zeroext cases.

// Bools are a bit of an odd case in CIR's x86_64 representation: they are considered i8
// everywhere except in the function return/arguments, where they are considered i1. To
// match LLVM's behavior, we need to zero-extend them when passing them as arguments.

// CHECK: @_Z4Boolb(%arg0: !cir.bool {cir.zeroext} loc({{.+}})) -> (!cir.bool {cir.zeroext})
bool Bool(bool a) {
// CHECK:   cir.call @_Z4Boolb({{.+}}) : (!cir.bool) -> !cir.bool
  return Bool(a);
}

// CHECK: cir.func @_Z5UCharh(%arg0: !u8i {cir.zeroext} loc({{.+}})) -> (!u8i {cir.zeroext})
unsigned char UChar(unsigned char c) {
  // CHECK: cir.call @_Z5UCharh(%2) : (!u8i) -> !u8i
  return UChar(c);
}
// CHECK: cir.func @_Z6UShortt(%arg0: !u16i {cir.zeroext} loc({{.+}})) -> (!u16i {cir.zeroext})
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

/// Test call conv lowering for trivial signext cases. ///

// CHECK: cir.func @_Z4Chara(%arg0: !s8i {cir.signext} loc({{.+}})) -> (!s8i {cir.signext})
char Char(signed char c) {
  // CHECK: cir.call @_Z4Chara(%{{.+}}) : (!s8i) -> !s8i
  return Char(c);
}
// CHECK: cir.func @_Z5Shorts(%arg0: !s16i {cir.signext} loc({{.+}})) -> (!s16i {cir.signext})
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


/// Test call conv lowering for struct type coercion scenarios. ///

struct S1 {
  int a, b;
};


/// Validate coerced argument and cast it to the expected type.

/// Cast arguments to the expected type.
// CHECK: %[[#V0:]] = cir.alloca !ty_S1_, !cir.ptr<!ty_S1_>, [""] {alignment = 4 : i64}
// CHECK: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S1_>), !cir.ptr<!u64i>
// CHECK: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CHECK: %[[#V2:]] = cir.alloca !ty_S1_, !cir.ptr<!ty_S1_>, ["__retval"] {alignment = 4 : i64}
// CHECK: %[[#V3:]] = cir.alloca !ty_S1_, !cir.ptr<!ty_S1_>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK: %[[#V4:]] = cir.alloca !ty_S1_, !cir.ptr<!ty_S1_>, ["agg.tmp1"] {alignment = 4 : i64}
S1 s1(S1 arg) {

  /// Cast argument and result of the function call to the expected types.
  // CHECK: %[[#V9:]] = cir.cast(bitcast, %[[#V3]] : !cir.ptr<!ty_S1_>), !cir.ptr<!u64i>
  // CHECK: %[[#V10:]] = cir.load %[[#V9]] : !cir.ptr<!u64i>, !u64i
  // CHECK: %[[#V11:]] = cir.call @_Z2s12S1(%[[#V10]]) : (!u64i) -> !u64i
  // CHECK: %[[#V12:]] = cir.cast(bitcast, %[[#V4]] : !cir.ptr<!ty_S1_>), !cir.ptr<!u64i>
  // CHECK: cir.store %[[#V11]], %[[#V12]] : !u64i, !cir.ptr<!u64i>
  s1({1, 2});

  // CHECK: %[[#V13:]] = cir.get_member %[[#V2]][0] {name = "a"} : !cir.ptr<!ty_S1_> -> !cir.ptr<!s32i>
  // CHECK: %[[#V14:]] = cir.const #cir.int<1> : !s32i
  // CHECK: cir.store %[[#V14]], %[[#V13]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[#V15:]] = cir.get_member %[[#V2]][1] {name = "b"} : !cir.ptr<!ty_S1_> -> !cir.ptr<!s32i>
  // CHECK: %[[#V16:]] = cir.const #cir.int<2> : !s32i
  // CHECK: cir.store %[[#V16]], %[[#V15]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[#V17:]] = cir.cast(bitcast, %[[#V2]] : !cir.ptr<!ty_S1_>), !cir.ptr<!u64i>
  // CHECK: %[[#V18:]] = cir.load %[[#V17]] : !cir.ptr<!u64i>, !u64i
  // CHECK: cir.return %[[#V18]] : !u64i
  return {1, 2};
}

/// Test call conv lowering for flattened structs. ///

struct S2 {
  int x, y, z;
};

// COM: Function prologue

// CHECK: cir.func @_Z2s22S2(%[[ARG0:[a-z0-9]+]]: !u64i {{.*}}, %[[ARG1:[a-z0-9]+]]: !s32i {{.*}}) -> !ty_anon_struct
// CHECK: %[[#F0:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>
// CHECK: %[[#F1:]] = cir.alloca !ty_anon_struct, !cir.ptr<!ty_anon_struct>
// CHECK: %[[#F2:]] = cir.get_member %[[#F1]][0]{{.*}} : !cir.ptr<!ty_anon_struct> -> !cir.ptr<!u64i>
// CHECK: cir.store %[[ARG0]], %[[#F2]] : !u64i, !cir.ptr<!u64i>
// CHECK: %[[#F3:]] = cir.get_member %[[#F1]][1]{{.*}} : !cir.ptr<!ty_anon_struct> -> !cir.ptr<!s32i>
// CHECK: cir.store %[[ARG1]], %[[#F3]] : !s32i, !cir.ptr<!s32i>
// CHECK: %[[#F4:]] = cir.cast(bitcast, %[[#F1]] : !cir.ptr<!ty_anon_struct>), !cir.ptr<!void>
// CHECK: %[[#F5:]] = cir.cast(bitcast, %[[#F0]] : !cir.ptr<!ty_S2_>), !cir.ptr<!void>
// CHECK: %[[#F6:]] = cir.const #cir.int<12> : !u64i
// CHECK: cir.libc.memcpy %[[#F6]] bytes from %[[#F4]] to %[[#F5]]
S2 s2(S2 arg) {
  // CHECK: %[[#F7:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["__retval"] {alignment = 4 : i64}
  // CHECK: %[[#F8:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["agg.tmp0"] {alignment = 4 : i64}
  // CHECK: %[[#F9:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["agg.tmp1"] {alignment = 4 : i64}
  // CHECK: %[[#F10:]] = cir.alloca !ty_anon_struct, !cir.ptr<!ty_anon_struct>, ["tmp"] {alignment = 8 : i64}
  // CHECK: %[[#F11:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["tmp"] {alignment = 4 : i64}
  // CHECK: %[[#F12:]] = cir.alloca !ty_anon_struct, !cir.ptr<!ty_anon_struct>, ["tmp"] {alignment = 8 : i64}
  // CHECK: %[[#F13:]] = cir.alloca !ty_anon_struct, !cir.ptr<!ty_anon_struct>, ["tmp"] {alignment = 8 : i64}
  
  // COM: Construction of S2 { 1, 2, 3 }.

  // CHECK: %[[#F14:]] = cir.get_member %[[#F8]][0] {{.*}} : !cir.ptr<!ty_S2_> -> !cir.ptr<!s32i>
  // CHECK: %[[#F15:]] = cir.const #cir.int<1> : !s32i
  // CHECK: cir.store %[[#F15]], %[[#F14]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[#F16:]] = cir.get_member %[[#F8]][1] {{.*}} : !cir.ptr<!ty_S2_> -> !cir.ptr<!s32i>
  // CHECK: %[[#F17:]] = cir.const #cir.int<2> : !s32i
  // CHECK: cir.store %[[#F17]], %[[#F16]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[#F18:]] = cir.get_member %[[#F8]][2] {{.*}} : !cir.ptr<!ty_S2_> -> !cir.ptr<!s32i>
  // CHECK: %[[#F19:]] = cir.const #cir.int<3> : !s32i
  // CHECK: cir.store %[[#F19]], %[[#F18]] : !s32i, !cir.ptr<!s32i>

  // COM: Flattening of the struct.
  // COM: { i32, i32, i32 } -> { i64, i32 }.

  // CHECK: %[[#F20:]] = cir.load %[[#F8]] : !cir.ptr<!ty_S2_>, !ty_S2_
  // CHECK: cir.store %[[#F20]], %[[#F11]] : !ty_S2_, !cir.ptr<!ty_S2_>
  // CHECK: %[[#F21:]] = cir.cast(bitcast, %[[#F11]] : !cir.ptr<!ty_S2_>), !cir.ptr<!void>
  // CHECK: %[[#F22:]] = cir.cast(bitcast, %[[#F10]] : !cir.ptr<!ty_anon_struct>), !cir.ptr<!void>
  // CHECK: %[[#F23:]] = cir.const #cir.int<12> : !u64i
  // CHECK: cir.libc.memcpy %[[#F23]] bytes from %[[#F21]] to %[[#F22]]

  // COM: Function call.
  // COM: Retrieve the two values in { i64, i32 }.

  // CHECK: %[[#F24:]] = cir.get_member %[[#F10]][0] {name = ""} : !cir.ptr<!ty_anon_struct> -> !cir.ptr<!u64i>
  // CHECK: %[[#F25:]] = cir.load %[[#F24]] : !cir.ptr<!u64i>, !u64i
  // CHECK: %[[#F26:]] = cir.get_member %[[#F10]][1] {name = ""} : !cir.ptr<!ty_anon_struct> -> !cir.ptr<!s32i>
  // CHECK: %[[#F27:]] = cir.load %[[#F26]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[#F28:]] = cir.call @_Z2s22S2(%[[#F25]], %[[#F27]]) : (!u64i, !s32i) -> !ty_anon_struct
  // CHECK: cir.store %[[#F28]], %[[#F12]] : !ty_anon_struct, !cir.ptr<!ty_anon_struct>

  // CHECK: %[[#F29:]] = cir.cast(bitcast, %[[#F12]] : !cir.ptr<!ty_anon_struct>), !cir.ptr<!void>
  // CHECK: %[[#F30:]] = cir.cast(bitcast, %[[#F9]] : !cir.ptr<!ty_S2_>), !cir.ptr<!void>
  // CHECK: %[[#F31:]] = cir.const #cir.int<12> : !u64i
  // CHECK: cir.libc.memcpy %[[#F31]] bytes from %[[#F29]] to %[[#F30]]

  // COM: Construct S2 { 1, 2, 3 } again.
  // COM: It has been tested above, so no duplication here.

  // COM: For return, the first two fields of S2 is also coerced.

  // CHECK: %[[#F39:]] = cir.cast(bitcast, %[[#F7]] : !cir.ptr<!ty_S2_>), !cir.ptr<!void>
  // CHECK: %[[#F40:]] = cir.cast(bitcast, %[[#F13]] : !cir.ptr<!ty_anon_struct>), !cir.ptr<!void>
  // CHECK: %[[#F41:]] = cir.const #cir.int<12> : !u64i
  // cir.libc.memcpy %[[#F41]] bytes from %[[#F39]] to %[[#F40]]
  // CHECK: %[[#F42:]] = cir.load %[[#F13]] : !cir.ptr<!ty_anon_struct>, !ty_anon_struct
  // cir.return %[[#F42]] : !ty_anon_struct
  s2({ 1, 2, 3 });
  return { 1, 2, 3 };
}
