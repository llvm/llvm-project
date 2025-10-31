// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  int X;
  float Y;
};

// struct truncation to a scalar
// CHECK-LABEL: define void {{.*}}call0
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[A]], align 4
export void call0() {
  S s = {1,2};
  int A = (int)s;
}

// struct from vector
// CHECK-LABEL: define void {{.*}}call1
// CHECK: [[A:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: store <2 x i32> <i32 1, i32 2>, ptr [[A]], align 8
// CHECK-NEXT: [[L:%.*]] = load <2 x i32>, ptr [[A]], align 8
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[VL:%.*]] = extractelement <2 x i32> [[L]], i64 0
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: [[VL2:%.*]] = extractelement <2 x i32> [[L]], i64 1
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[VL2]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call1() {
  int2 A = {1,2};
  S s = (S)A;
}


// struct from array
// CHECK-LABEL: define void {{.*}}call2
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L4]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call2() {
  int A[2] = {1,2};
  S s = (S)A;
}

struct Q {
  int Z;
};

struct R {
  Q q;
  float F;
};

// struct from nested struct?
// CHECK-LABEL: define void {{.*}}call6
// CHECK: [[r:%.*]] = alloca %struct.R, align 1
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.R, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[r]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[r]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds %struct.R, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds %struct.R, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load float, ptr [[G4]], align 4
// CHECK-NEXT: store float [[L4]], ptr [[G2]], align 4
export void call6() {
  R r = {{1}, 2.0};
  S s = (S)r;
}

// nested struct from array?
// CHECK-LABEL: define void {{.*}}call7
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[r:%.*]] = alloca %struct.R, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.R, ptr [[r]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.R, ptr [[r]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L4]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call7() {
  int A[2] = {1,2};
  R r = (R)A;
}

struct T {
  int A;
  int B;
  int C;
};

// struct truncation
// CHECK-LABEL: define void {{.*}}call8
// CHECK: [[t:%.*]] = alloca %struct.T, align 1
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.T, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[t]], ptr align 1 {{.*}}, i32 12, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[t]], i32 12, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds %struct.T, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds %struct.T, ptr %agg-temp, i32 0, i32 1
// CHECK-NEXT: [[G5:%.*]] = getelementptr inbounds %struct.T, ptr %agg-temp, i32 0, i32 2
// CHECK-NEXT: [[L1:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L1]], ptr [[G1]], align 4
// CHECK-NEXT: [[L2:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L2]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call8() {
  T t = {1,2,3};
  S s = (S)t;
}

struct BFields {
  double D;
  int E: 15;
  int : 8;
  float F;
};

struct Derived : BFields {
  int G;
};

// Derived Struct truncate to scalar
// CHECK-LABEL: call9
// CHECK: [[D2:%.*]] = alloca double, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 %D, i32 19, i1 false)
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep]], i32 0, i32 1
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[A:%.*]] = load double, ptr [[Gep1]], align 8
// CHECK-NEXT: store double [[A]], ptr [[D2]], align 8
// CHECK-NEXT: ret void
export void call9(Derived D) {
  double D2 = (double)D;
}

// Derived struct from vector
// CHECK-LABEL: call10
// CHECK: [[IAddr:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[D:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: store <4 x i32> %I, ptr [[IAddr]], align 16
// CHECK-NEXT: [[A:%.*]] = load <4 x i32>, ptr [[IAddr]], align 16
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.Derived, ptr [[D]], i32 0, i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep]], i32 0, i32 1
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[D]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[D]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds %struct.Derived, ptr [[D]], i32 0, i32 1
// CHECK-NEXT: [[VL:%.*]] = extractelement <4 x i32> [[A]], i64 0
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[VL]] to double
// CHECK-NEXT: store double [[C]], ptr [[Gep1]], align 8
// CHECK-NEXT: [[VL4:%.*]] = extractelement <4 x i32> [[A]], i64 1
// CHECK-NEXT: [[B:%.*]] = trunc i32 [[VL4]] to i24
// CHECK-NEXT: [[BFL:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT: [[BFV:%.*]] = and i24 [[B]], 32767
// CHECK-NEXT: [[BFC:%.*]] = and i24 [[BFL]], -32768
// CHECK-NEXT: [[BFSet:%.*]] = or i24 [[BFC]], [[BFV]]
// CHECK-NEXT: store i24 [[BFSet]], ptr [[E]], align 1
// CHECK-NEXT: [[VL5:%.*]] = extractelement <4 x i32> [[A]], i64 2
// CHECK-NEXT: [[C6:%.*]] = sitofp i32 [[VL5]] to float
// CHECK-NEXT: store float [[C6]], ptr [[Gep2]], align 4
// CHECK-NEXT: [[VL7:%.*]] = extractelement <4 x i32> [[A]], i64 3
// CHECK-NEXT: store i32 [[VL7]], ptr [[Gep3]], align 4
// CHECK-NEXT: ret void
export void call10(int4 I) {
  Derived D = (Derived)I;
}

// truncate derived struct
// CHECK-LABEL: call11
// CHECK: [[B:%.*]] = alloca %struct.BFields, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[D]], i32 19, i1 false)
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.BFields, ptr [[B]], i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep]], i32 0, i32 1
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds %struct.BFields, ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds %struct.BFields, ptr [[B]], i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[E4:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep3]], i32 0, i32 1
// CHECK-NEXT: [[Gep5:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep6:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep7:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[A:%.*]] = load double, ptr [[Gep5]], align 8
// CHECK-NEXT: store double [[A]], ptr [[Gep1]], align 8
// CHECK-NEXT: [[BFl:%.*]] = load i24, ptr [[E4]], align 1
// CHECK-NEXT: [[Shl:%.*]] = shl i24 [[BFL]], 9
// CHECK-NEXT: [[Ashr:%.*]] = ashr i24 [[Shl]], 9
// CHECK-NEXT: [[BFC:%.*]] = sext i24 [[Ashr]] to i32
// CHECK-NEXT: [[B:%.*]] = trunc i32 [[BFC]] to i24
// CHECK-NEXT: [[BFL8:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT: [[BFV:%.*]] = and i24 [[B]], 32767
// CHECK-NEXT: [[BFC:%.*]] = and i24 [[BFL8]], -32768
// CHECK-NEXT: [[BFSet:%.*]] = or i24 [[BFC]], [[BFV]]
// CHECK-NEXT: store i24 [[BFSet]], ptr [[E]], align 1
// CHECK-NEXT: [[C:%.*]] = load float, ptr [[Gep6]], align 4
// CHECK-NEXT: store float [[C]], ptr [[Gep2]], align 4
// CHECK-NEXT: ret void
export void call11(Derived D) {
  BFields B = (BFields)D;
}

struct Empty {
};

// cast to an empty struct
// CHECK-LABEL: call12
// CHECK: [[I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[E:%.*]] = alloca %struct.Empty, align 1
// CHECK-NEXT: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr [[I]], align 16
// CHECK-NEXT: [[A:%.*]] = load <4 x i32>, ptr [[I]], align 16
// CHECK-NEXt: ret void
export void call12() {
  int4 I = {1,2,3,4};
  Empty E = (Empty)I;
}

struct MoreBFields {
  int A;
  uint64_t B: 60;
  float C;
  uint16_t D: 10;
  uint16_t E: 6;
  int : 32;
  double F;
  int : 8;
  uint G;
};

// more complicated bitfield case
// CHECK-LABEL: call13
// CHECK: [[AA:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[MBF:%.*]] = alloca %struct.MoreBFields, align 1
// CHECK-NEXT: store i32 %A, ptr [[AA]], align 4
// CHECK-NEXT: [[Z:%.*]] = load i32, ptr [[AA]], align 4
// get the gep for the struct.
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.MoreBFields, ptr [[MBF]], i32 0
// CHECK-NEXT: [[FieldB:%.*]] = getelementptr inbounds nuw %struct.MoreBFields, ptr [[Gep]], i32 0, i32 1
// D and E share the same field index
// CHECK-NEXT: [[FieldD:%.*]] = getelementptr inbounds nuw %struct.MoreBFields, ptr [[Gep]], i32 0, i32 3
// CHECK-NEXT: [[FieldE:%.*]] = getelementptr inbounds nuw %struct.MoreBFields, ptr [[Gep]], i32 0, i32 3
// CHECK-NEXT: [[FieldA:%.*]] = getelementptr inbounds %struct.MoreBFields, ptr [[MBF]], i32 0, i32 0
// CHECK-NEXT: [[FieldC:%.*]] = getelementptr inbounds %struct.MoreBFields, ptr [[MBF]], i32 0, i32 2
// CHECK-NEXT: [[FieldF:%.*]] = getelementptr inbounds %struct.MoreBFields, ptr [[MBF]], i32 0, i32 5
// CHECK-NEXT: [[FieldG:%.*]] = getelementptr inbounds %struct.MoreBFields, ptr [[MBF]], i32 0, i32 7
// store int A into field A
// CHECK-NEXT: store i32 [[Z]], ptr [[FieldA]], align 4
// store int A in bitField B, do necessary conversions
// CHECK-NEXT: [[Conv:%.*]] = sext i32 [[Z]] to i64
// CHECK-NEXT: [[BFL:%.*]] = load i64, ptr [[FieldB]], align 1
// CHECK-NEXT: [[BFV:%.*]] = and i64 [[Conv]], 1152921504606846975
// CHECK-NEXT: [[BFC:%.*]] = and i64 [[BFL]], -1152921504606846976
// CHECK-NEXT: [[BFS:%.*]] = or i64 [[BFC]], [[BFV]]
// CHECK-NEXT: store i64 [[BFS]], ptr [[FieldB]], align 1
// store int A into field C
// CHECK-NEXT: [[Conv5:%.*]] = sitofp i32 [[Z]] to float
// CHECK-NEXT: store float [[Conv5]], ptr [[FieldC]], align 4
// store int A into bitfield D
// CHECK-NEXT: [[Conv6:%.*]] = trunc i32 [[Z]] to i16
// CHECK-NEXT: [[FDL:%.*]] = load i16, ptr [[FieldD]], align 1
// CHECK-NEXT: [[FDV:%.*]] = and i16 [[Conv6]], 1023
// CHECK-NEXT: [[FDC:%.*]] = and i16 [[FDL]], -1024
// CHECK-NEXT: [[FDS:%.*]] = or i16 [[FDC]], [[FDV]]
// CHECK-NEXT: store i16 [[FDS]], ptr [[FieldD]], align 1
// store int A into bitfield E;
// CHECK-NEXT: [[Conv11:%.*]] = trunc i32 [[Z]] to i16
// CHECK-NEXT: [[FEL:%.*]] = load i16, ptr [[FieldE]], align 1
// CHECK-NEXT: [[FEV:%.*]] = and i16 [[Conv11]], 63
// CHECK-NEXT: [[FESHL:%.*]] = shl i16 [[FEV]], 10
// CHECK-NEXT: [[FEC:%.*]] = and i16 [[FEL]], 1023
// CHECK-NEXT: [[FES:%.*]] = or i16 [[FEC]], [[FESHL]]
// CHECK-NEXT: store i16 [[FES]], ptr [[FieldE]], align 1
// store int A into field F
// CHECK-NEXT: [[Conv16:%.*]] = sitofp i32 [[Z]] to double
// CHECK-NEXT: store double [[Conv16]], ptr [[FieldF]], align 8
// store int A into field G
// CHECK-NEXT: store i32 [[Z]], ptr [[FieldG]], align 4
// CHECK-NEXT: ret void
export void call13(int A) {
  MoreBFields MBF = (MoreBFields)A;
}

struct Inner {
  int Z;
  int Y : 25;
};

struct Outer {
  int A;
  Inner I;
};

// show usage of "extra" gep for struct containing bitfield
// CHECK-LABEL: call14
// CHECK: [[AA:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[O:%.*]] = alloca %struct.Outer, align 1
// CHECK-NEXT: store i32 %A, ptr [[AA]], align 4
// CHECK-NEXT: [[Z:%.*]] = load i32, ptr [[AA]], align 4
// CHECK-NEXT: [[FieldA:%.*]] = getelementptr inbounds %struct.Outer, ptr [[O]], i32 0, i32 0
// showing real usage of "extra gep". need Inner struct to generate access of its bitfield.
// CHECK-NEXT: [[FieldI:%.*]] = getelementptr inbounds %struct.Outer, ptr [[O]], i32 0, i32 1
// CHECK-NEXT: [[FieldY:%.*]] = getelementptr inbounds nuw %struct.Inner, ptr [[FieldI]], i32 0, i32 1
// CHECK-NEXT: [[FieldZ:%.*]] = getelementptr inbounds %struct.Outer, ptr [[O]], i32 0, i32 1, i32 0
// CHECK-NEXT: store i32 [[Z]], ptr [[FieldA]], align 4
// CHECK-NEXT: store i32 [[Z]], ptr [[FieldZ]], align 4
// CHECK-NEXT: [[BFL:%.*]] = load i32, ptr [[FieldY]], align 1
// CHECK-NEXT: [[BFV:%.*]] = and i32 [[Z]], 33554431
// CHECK-NEXT: [[BFC:%.*]] = and i32 [[BFL]], -33554432
// CHECK-NEXT: [[BFS:%.*]] = or i32 [[BFC]], [[BFV]]
// CHECK-NEXT: store i32 [[BFS]], ptr [[FieldY]], align 1
// CHECK-NEXT: ret void
export void call14(int A) {
  Outer O = (Outer)A;
}
