// RUN: %clang_cc1 -faltivec -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Test homogeneous float aggregate passing and returning.

struct f1 { float f[1]; };
struct f2 { float f[2]; };
struct f3 { float f[3]; };
struct f4 { float f[4]; };
struct f5 { float f[5]; };
struct f6 { float f[6]; };
struct f7 { float f[7]; };
struct f8 { float f[8]; };
struct f9 { float f[9]; };

struct fab { float a; float b; };
struct fabc { float a; float b; float c; };

struct f2a2b { float a[2]; float b[2]; };

// CHECK: define [1 x float] @func_f1(float inreg %x.coerce)
struct f1 func_f1(struct f1 x) { return x; }

// CHECK: define [2 x float] @func_f2([2 x float] %x.coerce)
struct f2 func_f2(struct f2 x) { return x; }

// CHECK: define [3 x float] @func_f3([3 x float] %x.coerce)
struct f3 func_f3(struct f3 x) { return x; }

// CHECK: define [4 x float] @func_f4([4 x float] %x.coerce)
struct f4 func_f4(struct f4 x) { return x; }

// CHECK: define [5 x float] @func_f5([5 x float] %x.coerce)
struct f5 func_f5(struct f5 x) { return x; }

// CHECK: define [6 x float] @func_f6([6 x float] %x.coerce)
struct f6 func_f6(struct f6 x) { return x; }

// CHECK: define [7 x float] @func_f7([7 x float] %x.coerce)
struct f7 func_f7(struct f7 x) { return x; }

// CHECK: define [8 x float] @func_f8([8 x float] %x.coerce)
struct f8 func_f8(struct f8 x) { return x; }

// CHECK: define void @func_f9(%struct.f9* noalias sret %agg.result, [5 x i64] %x.coerce)
struct f9 func_f9(struct f9 x) { return x; }

// CHECK: define [2 x float] @func_fab([2 x float] %x.coerce)
struct fab func_fab(struct fab x) { return x; }

// CHECK: define [3 x float] @func_fabc([3 x float] %x.coerce)
struct fabc func_fabc(struct fabc x) { return x; }

// CHECK: define [4 x float] @func_f2a2b([4 x float] %x.coerce)
struct f2a2b func_f2a2b(struct f2a2b x) { return x; }

// CHECK-LABEL: @call_f1
// CHECK: %[[TMP:[^ ]+]] = load float, float* getelementptr inbounds (%struct.f1, %struct.f1* @global_f1, i32 0, i32 0, i32 0), align 4
// CHECK: call [1 x float] @func_f1(float inreg %[[TMP]])
struct f1 global_f1;
void call_f1(void) { global_f1 = func_f1(global_f1); }

// CHECK-LABEL: @call_f2
// CHECK: %[[TMP:[^ ]+]] = load [2 x float], [2 x float]* getelementptr inbounds (%struct.f2, %struct.f2* @global_f2, i32 0, i32 0), align 4
// CHECK: call [2 x float] @func_f2([2 x float] %[[TMP]])
struct f2 global_f2;
void call_f2(void) { global_f2 = func_f2(global_f2); }

// CHECK-LABEL: @call_f3
// CHECK: %[[TMP:[^ ]+]] = load [3 x float], [3 x float]* getelementptr inbounds (%struct.f3, %struct.f3* @global_f3, i32 0, i32 0), align 4
// CHECK: call [3 x float] @func_f3([3 x float] %[[TMP]])
struct f3 global_f3;
void call_f3(void) { global_f3 = func_f3(global_f3); }

// CHECK-LABEL: @call_f4
// CHECK: %[[TMP:[^ ]+]] = load [4 x float], [4 x float]* getelementptr inbounds (%struct.f4, %struct.f4* @global_f4, i32 0, i32 0), align 4
// CHECK: call [4 x float] @func_f4([4 x float] %[[TMP]])
struct f4 global_f4;
void call_f4(void) { global_f4 = func_f4(global_f4); }

// CHECK-LABEL: @call_f5
// CHECK: %[[TMP:[^ ]+]] = load [5 x float], [5 x float]* getelementptr inbounds (%struct.f5, %struct.f5* @global_f5, i32 0, i32 0), align 4
// CHECK: call [5 x float] @func_f5([5 x float] %[[TMP]])
struct f5 global_f5;
void call_f5(void) { global_f5 = func_f5(global_f5); }

// CHECK-LABEL: @call_f6
// CHECK: %[[TMP:[^ ]+]] = load [6 x float], [6 x float]* getelementptr inbounds (%struct.f6, %struct.f6* @global_f6, i32 0, i32 0), align 4
// CHECK: call [6 x float] @func_f6([6 x float] %[[TMP]])
struct f6 global_f6;
void call_f6(void) { global_f6 = func_f6(global_f6); }

// CHECK-LABEL: @call_f7
// CHECK: %[[TMP:[^ ]+]] = load [7 x float], [7 x float]* getelementptr inbounds (%struct.f7, %struct.f7* @global_f7, i32 0, i32 0), align 4
// CHECK: call [7 x float] @func_f7([7 x float] %[[TMP]])
struct f7 global_f7;
void call_f7(void) { global_f7 = func_f7(global_f7); }

// CHECK-LABEL: @call_f8
// CHECK: %[[TMP:[^ ]+]] = load [8 x float], [8 x float]* getelementptr inbounds (%struct.f8, %struct.f8* @global_f8, i32 0, i32 0), align 4
// CHECK: call [8 x float] @func_f8([8 x float] %[[TMP]])
struct f8 global_f8;
void call_f8(void) { global_f8 = func_f8(global_f8); }

// CHECK-LABEL: @call_f9
// CHECK: %[[TMP1:[^ ]+]] = alloca [5 x i64]
// CHECK: %[[TMP2:[^ ]+]] = bitcast [5 x i64]* %[[TMP1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %[[TMP2]], i8* align 4 bitcast (%struct.f9* @global_f9 to i8*), i64 36, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [5 x i64], [5 x i64]* %[[TMP1]]
// CHECK: call void @func_f9(%struct.f9* sret %{{[^ ]+}}, [5 x i64] %[[TMP3]])
struct f9 global_f9;
void call_f9(void) { global_f9 = func_f9(global_f9); }

// CHECK-LABEL: @call_fab
// CHECK: %[[TMP:[^ ]+]] = load [2 x float], [2 x float]* bitcast (%struct.fab* @global_fab to [2 x float]*)
// CHECK: call [2 x float] @func_fab([2 x float] %[[TMP]])
struct fab global_fab;
void call_fab(void) { global_fab = func_fab(global_fab); }

// CHECK-LABEL: @call_fabc
// CHECK: %[[TMP:[^ ]+]] = load [3 x float], [3 x float]* bitcast (%struct.fabc* @global_fabc to [3 x float]*)
// CHECK: call [3 x float] @func_fabc([3 x float] %[[TMP]])
struct fabc global_fabc;
void call_fabc(void) { global_fabc = func_fabc(global_fabc); }


// Test homogeneous vector aggregate passing and returning.

struct v1 { vector int v[1]; };
struct v2 { vector int v[2]; };
struct v3 { vector int v[3]; };
struct v4 { vector int v[4]; };
struct v5 { vector int v[5]; };
struct v6 { vector int v[6]; };
struct v7 { vector int v[7]; };
struct v8 { vector int v[8]; };
struct v9 { vector int v[9]; };

struct vab { vector int a; vector int b; };
struct vabc { vector int a; vector int b; vector int c; };

// CHECK: define [1 x <4 x i32>] @func_v1(<4 x i32> inreg %x.coerce)
struct v1 func_v1(struct v1 x) { return x; }

// CHECK: define [2 x <4 x i32>] @func_v2([2 x <4 x i32>] %x.coerce)
struct v2 func_v2(struct v2 x) { return x; }

// CHECK: define [3 x <4 x i32>] @func_v3([3 x <4 x i32>] %x.coerce)
struct v3 func_v3(struct v3 x) { return x; }

// CHECK: define [4 x <4 x i32>] @func_v4([4 x <4 x i32>] %x.coerce)
struct v4 func_v4(struct v4 x) { return x; }

// CHECK: define [5 x <4 x i32>] @func_v5([5 x <4 x i32>] %x.coerce)
struct v5 func_v5(struct v5 x) { return x; }

// CHECK: define [6 x <4 x i32>] @func_v6([6 x <4 x i32>] %x.coerce)
struct v6 func_v6(struct v6 x) { return x; }

// CHECK: define [7 x <4 x i32>] @func_v7([7 x <4 x i32>] %x.coerce)
struct v7 func_v7(struct v7 x) { return x; }

// CHECK: define [8 x <4 x i32>] @func_v8([8 x <4 x i32>] %x.coerce)
struct v8 func_v8(struct v8 x) { return x; }

// CHECK: define void @func_v9(%struct.v9* noalias sret %agg.result, %struct.v9* byval align 16 %x)
struct v9 func_v9(struct v9 x) { return x; }

// CHECK: define [2 x <4 x i32>] @func_vab([2 x <4 x i32>] %x.coerce)
struct vab func_vab(struct vab x) { return x; }

// CHECK: define [3 x <4 x i32>] @func_vabc([3 x <4 x i32>] %x.coerce)
struct vabc func_vabc(struct vabc x) { return x; }

// CHECK-LABEL: @call_v1
// CHECK: %[[TMP:[^ ]+]] = load <4 x i32>, <4 x i32>* getelementptr inbounds (%struct.v1, %struct.v1* @global_v1, i32 0, i32 0, i32 0), align 1
// CHECK: call [1 x <4 x i32>] @func_v1(<4 x i32> inreg %[[TMP]])
struct v1 global_v1;
void call_v1(void) { global_v1 = func_v1(global_v1); }

// CHECK-LABEL: @call_v2
// CHECK: %[[TMP:[^ ]+]] = load [2 x <4 x i32>], [2 x <4 x i32>]* getelementptr inbounds (%struct.v2, %struct.v2* @global_v2, i32 0, i32 0), align 1
// CHECK: call [2 x <4 x i32>] @func_v2([2 x <4 x i32>] %[[TMP]])
struct v2 global_v2;
void call_v2(void) { global_v2 = func_v2(global_v2); }

// CHECK-LABEL: @call_v3
// CHECK: %[[TMP:[^ ]+]] = load [3 x <4 x i32>], [3 x <4 x i32>]* getelementptr inbounds (%struct.v3, %struct.v3* @global_v3, i32 0, i32 0), align 1
// CHECK: call [3 x <4 x i32>] @func_v3([3 x <4 x i32>] %[[TMP]])
struct v3 global_v3;
void call_v3(void) { global_v3 = func_v3(global_v3); }

// CHECK-LABEL: @call_v4
// CHECK: %[[TMP:[^ ]+]] = load [4 x <4 x i32>], [4 x <4 x i32>]* getelementptr inbounds (%struct.v4, %struct.v4* @global_v4, i32 0, i32 0), align 1
// CHECK: call [4 x <4 x i32>] @func_v4([4 x <4 x i32>] %[[TMP]])
struct v4 global_v4;
void call_v4(void) { global_v4 = func_v4(global_v4); }

// CHECK-LABEL: @call_v5
// CHECK: %[[TMP:[^ ]+]] = load [5 x <4 x i32>], [5 x <4 x i32>]* getelementptr inbounds (%struct.v5, %struct.v5* @global_v5, i32 0, i32 0), align 1
// CHECK: call [5 x <4 x i32>] @func_v5([5 x <4 x i32>] %[[TMP]])
struct v5 global_v5;
void call_v5(void) { global_v5 = func_v5(global_v5); }

// CHECK-LABEL: @call_v6
// CHECK: %[[TMP:[^ ]+]] = load [6 x <4 x i32>], [6 x <4 x i32>]* getelementptr inbounds (%struct.v6, %struct.v6* @global_v6, i32 0, i32 0), align 1
// CHECK: call [6 x <4 x i32>] @func_v6([6 x <4 x i32>] %[[TMP]])
struct v6 global_v6;
void call_v6(void) { global_v6 = func_v6(global_v6); }

// CHECK-LABEL: @call_v7
// CHECK: %[[TMP:[^ ]+]] = load [7 x <4 x i32>], [7 x <4 x i32>]* getelementptr inbounds (%struct.v7, %struct.v7* @global_v7, i32 0, i32 0), align 1
// CHECK: call [7 x <4 x i32>] @func_v7([7 x <4 x i32>] %[[TMP]])
struct v7 global_v7;
void call_v7(void) { global_v7 = func_v7(global_v7); }

// CHECK-LABEL: @call_v8
// CHECK: %[[TMP:[^ ]+]] = load [8 x <4 x i32>], [8 x <4 x i32>]* getelementptr inbounds (%struct.v8, %struct.v8* @global_v8, i32 0, i32 0), align 1
// CHECK: call [8 x <4 x i32>] @func_v8([8 x <4 x i32>] %[[TMP]])
struct v8 global_v8;
void call_v8(void) { global_v8 = func_v8(global_v8); }

// CHECK-LABEL: @call_v9
// CHECK: call void @func_v9(%struct.v9* sret %{{[^ ]+}}, %struct.v9* byval align 16 @global_v9)
struct v9 global_v9;
void call_v9(void) { global_v9 = func_v9(global_v9); }

// CHECK-LABEL: @call_vab
// CHECK: %[[TMP:[^ ]+]] = load [2 x <4 x i32>], [2 x <4 x i32>]* bitcast (%struct.vab* @global_vab to [2 x <4 x i32>]*)
// CHECK: call [2 x <4 x i32>] @func_vab([2 x <4 x i32>] %[[TMP]])
struct vab global_vab;
void call_vab(void) { global_vab = func_vab(global_vab); }

// CHECK-LABEL: @call_vabc
// CHECK: %[[TMP:[^ ]+]] = load [3 x <4 x i32>], [3 x <4 x i32>]* bitcast (%struct.vabc* @global_vabc to [3 x <4 x i32>]*)
// CHECK: call [3 x <4 x i32>] @func_vabc([3 x <4 x i32>] %[[TMP]])
struct vabc global_vabc;
void call_vabc(void) { global_vabc = func_vabc(global_vabc); }


// As clang extension, non-power-of-two vectors may also be part of
// homogeneous aggregates.

typedef float float3 __attribute__((vector_size (12)));

struct v3f1 { float3 v[1]; };
struct v3f2 { float3 v[2]; };
struct v3f3 { float3 v[3]; };
struct v3f4 { float3 v[4]; };
struct v3f5 { float3 v[5]; };
struct v3f6 { float3 v[6]; };
struct v3f7 { float3 v[7]; };
struct v3f8 { float3 v[8]; };
struct v3f9 { float3 v[9]; };

struct v3fab { float3 a; float3 b; };
struct v3fabc { float3 a; float3 b; float3 c; };

// CHECK: define [1 x <3 x float>] @func_v3f1(<3 x float> inreg %x.coerce)
struct v3f1 func_v3f1(struct v3f1 x) { return x; }

// CHECK: define [2 x <3 x float>] @func_v3f2([2 x <3 x float>] %x.coerce)
struct v3f2 func_v3f2(struct v3f2 x) { return x; }

// CHECK: define [3 x <3 x float>] @func_v3f3([3 x <3 x float>] %x.coerce)
struct v3f3 func_v3f3(struct v3f3 x) { return x; }

// CHECK: define [4 x <3 x float>] @func_v3f4([4 x <3 x float>] %x.coerce)
struct v3f4 func_v3f4(struct v3f4 x) { return x; }

// CHECK: define [5 x <3 x float>] @func_v3f5([5 x <3 x float>] %x.coerce)
struct v3f5 func_v3f5(struct v3f5 x) { return x; }

// CHECK: define [6 x <3 x float>] @func_v3f6([6 x <3 x float>] %x.coerce)
struct v3f6 func_v3f6(struct v3f6 x) { return x; }

// CHECK: define [7 x <3 x float>] @func_v3f7([7 x <3 x float>] %x.coerce)
struct v3f7 func_v3f7(struct v3f7 x) { return x; }

// CHECK: define [8 x <3 x float>] @func_v3f8([8 x <3 x float>] %x.coerce)
struct v3f8 func_v3f8(struct v3f8 x) { return x; }

// CHECK: define void @func_v3f9(%struct.v3f9* noalias sret %agg.result, %struct.v3f9* byval align 16 %x)
struct v3f9 func_v3f9(struct v3f9 x) { return x; }

// CHECK: define [2 x <3 x float>] @func_v3fab([2 x <3 x float>] %x.coerce)
struct v3fab func_v3fab(struct v3fab x) { return x; }

// CHECK: define [3 x <3 x float>] @func_v3fabc([3 x <3 x float>] %x.coerce)
struct v3fabc func_v3fabc(struct v3fabc x) { return x; }

// CHECK-LABEL: @call_v3f1
// CHECK: %[[TMP:[^ ]+]] = load <3 x float>, <3 x float>* getelementptr inbounds (%struct.v3f1, %struct.v3f1* @global_v3f1, i32 0, i32 0, i32 0), align 1
// CHECK: call [1 x <3 x float>] @func_v3f1(<3 x float> inreg %[[TMP]])
struct v3f1 global_v3f1;
void call_v3f1(void) { global_v3f1 = func_v3f1(global_v3f1); }

// CHECK-LABEL: @call_v3f2
// CHECK: %[[TMP:[^ ]+]] = load [2 x <3 x float>], [2 x <3 x float>]* getelementptr inbounds (%struct.v3f2, %struct.v3f2* @global_v3f2, i32 0, i32 0), align 1
// CHECK: call [2 x <3 x float>] @func_v3f2([2 x <3 x float>] %[[TMP]])
struct v3f2 global_v3f2;
void call_v3f2(void) { global_v3f2 = func_v3f2(global_v3f2); }

// CHECK-LABEL: @call_v3f3
// CHECK: %[[TMP:[^ ]+]] = load [3 x <3 x float>], [3 x <3 x float>]* getelementptr inbounds (%struct.v3f3, %struct.v3f3* @global_v3f3, i32 0, i32 0), align 1
// CHECK: call [3 x <3 x float>] @func_v3f3([3 x <3 x float>] %[[TMP]])
struct v3f3 global_v3f3;
void call_v3f3(void) { global_v3f3 = func_v3f3(global_v3f3); }

// CHECK-LABEL: @call_v3f4
// CHECK: %[[TMP:[^ ]+]] = load [4 x <3 x float>], [4 x <3 x float>]* getelementptr inbounds (%struct.v3f4, %struct.v3f4* @global_v3f4, i32 0, i32 0), align 1
// CHECK: call [4 x <3 x float>] @func_v3f4([4 x <3 x float>] %[[TMP]])
struct v3f4 global_v3f4;
void call_v3f4(void) { global_v3f4 = func_v3f4(global_v3f4); }

// CHECK-LABEL: @call_v3f5
// CHECK: %[[TMP:[^ ]+]] = load [5 x <3 x float>], [5 x <3 x float>]* getelementptr inbounds (%struct.v3f5, %struct.v3f5* @global_v3f5, i32 0, i32 0), align 1
// CHECK: call [5 x <3 x float>] @func_v3f5([5 x <3 x float>] %[[TMP]])
struct v3f5 global_v3f5;
void call_v3f5(void) { global_v3f5 = func_v3f5(global_v3f5); }

// CHECK-LABEL: @call_v3f6
// CHECK: %[[TMP:[^ ]+]] = load [6 x <3 x float>], [6 x <3 x float>]* getelementptr inbounds (%struct.v3f6, %struct.v3f6* @global_v3f6, i32 0, i32 0), align 1
// CHECK: call [6 x <3 x float>] @func_v3f6([6 x <3 x float>] %[[TMP]])
struct v3f6 global_v3f6;
void call_v3f6(void) { global_v3f6 = func_v3f6(global_v3f6); }

// CHECK-LABEL: @call_v3f7
// CHECK: %[[TMP:[^ ]+]] = load [7 x <3 x float>], [7 x <3 x float>]* getelementptr inbounds (%struct.v3f7, %struct.v3f7* @global_v3f7, i32 0, i32 0), align 1
// CHECK: call [7 x <3 x float>] @func_v3f7([7 x <3 x float>] %[[TMP]])
struct v3f7 global_v3f7;
void call_v3f7(void) { global_v3f7 = func_v3f7(global_v3f7); }

// CHECK-LABEL: @call_v3f8
// CHECK: %[[TMP:[^ ]+]] = load [8 x <3 x float>], [8 x <3 x float>]* getelementptr inbounds (%struct.v3f8, %struct.v3f8* @global_v3f8, i32 0, i32 0), align 1
// CHECK: call [8 x <3 x float>] @func_v3f8([8 x <3 x float>] %[[TMP]])
struct v3f8 global_v3f8;
void call_v3f8(void) { global_v3f8 = func_v3f8(global_v3f8); }

// CHECK-LABEL: @call_v3f9
// CHECK: call void @func_v3f9(%struct.v3f9* sret %{{[^ ]+}}, %struct.v3f9* byval align 16 @global_v3f9)
struct v3f9 global_v3f9;
void call_v3f9(void) { global_v3f9 = func_v3f9(global_v3f9); }

// CHECK-LABEL: @call_v3fab
// CHECK: %[[TMP:[^ ]+]] = load [2 x <3 x float>], [2 x <3 x float>]* bitcast (%struct.v3fab* @global_v3fab to [2 x <3 x float>]*)
// CHECK: call [2 x <3 x float>] @func_v3fab([2 x <3 x float>] %[[TMP]])
struct v3fab global_v3fab;
void call_v3fab(void) { global_v3fab = func_v3fab(global_v3fab); }

// CHECK-LABEL: @call_v3fabc
// CHECK: %[[TMP:[^ ]+]] = load [3 x <3 x float>], [3 x <3 x float>]* bitcast (%struct.v3fabc* @global_v3fabc to [3 x <3 x float>]*)
// CHECK: call [3 x <3 x float>] @func_v3fabc([3 x <3 x float>] %[[TMP]])
struct v3fabc global_v3fabc;
void call_v3fabc(void) { global_v3fabc = func_v3fabc(global_v3fabc); }


// Test returning small aggregates.

struct s1 { char c[1]; };
struct s2 { char c[2]; };
struct s3 { char c[3]; };
struct s4 { char c[4]; };
struct s5 { char c[5]; };
struct s6 { char c[6]; };
struct s7 { char c[7]; };
struct s8 { char c[8]; };
struct s9 { char c[9]; };
struct s16 { char c[16]; };
struct s17 { char c[17]; };

// CHECK: define i8 @ret_s1()
struct s1 ret_s1() {
  return (struct s1) { 17 };
}

// CHECK: define i16 @ret_s2()
struct s2 ret_s2() {
  return (struct s2) { 17, 18 };
}

// CHECK: define i24 @ret_s3()
struct s3 ret_s3() {
  return (struct s3) { 17, 18, 19 };
}

// CHECK: define i32 @ret_s4()
struct s4 ret_s4() {
  return (struct s4) { 17, 18, 19, 20 };
}

// CHECK: define i40 @ret_s5()
struct s5 ret_s5() {
  return (struct s5) { 17, 18, 19, 20, 21 };
}

// CHECK: define i48 @ret_s6()
struct s6 ret_s6() {
  return (struct s6) { 17, 18, 19, 20, 21, 22 };
}

// CHECK: define i56 @ret_s7()
struct s7 ret_s7() {
  return (struct s7) { 17, 18, 19, 20, 21, 22, 23 };
}

// CHECK: define i64 @ret_s8()
struct s8 ret_s8() {
  return (struct s8) { 17, 18, 19, 20, 21, 22, 23, 24 };
}

// CHECK: define { i64, i64 } @ret_s9()
struct s9 ret_s9() {
  return (struct s9) { 17, 18, 19, 20, 21, 22, 23, 24, 25 };
}

// CHECK: define { i64, i64 } @ret_s16()
struct s16 ret_s16() {
  return (struct s16) { 17, 18, 19, 20, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31, 32 };
}

// CHECK: define void @ret_s17(%struct.s17*
struct s17 ret_s17() {
  return (struct s17) { 17, 18, 19, 20, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31, 32, 33 };
}

