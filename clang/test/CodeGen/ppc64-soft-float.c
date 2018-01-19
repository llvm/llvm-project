// RUN: %clang_cc1 -msoft-float -mfloat-abi soft -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-LE %s
// RUN: %clang_cc1 -msoft-float -mfloat-abi soft -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-BE %s

// Test float returns and params.

// CHECK: define float @func_p1(float %x)
float func_p1(float x) { return x; }

// CHECK: define double @func_p2(double %x)
double func_p2(double x) { return x; }

// CHECK: define ppc_fp128 @func_p3(ppc_fp128 %x)
long double func_p3(long double x) { return x; }

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

// CHECK-LE: define i32 @func_f1(float inreg %x.coerce)
// CHECK-BE: define void @func_f1(%struct.f1* noalias sret %agg.result, float inreg %x.coerce)
struct f1 func_f1(struct f1 x) { return x; }

// CHECK-LE: define i64 @func_f2(i64 %x.coerce)
// CHECK-BE: define void @func_f2(%struct.f2* noalias sret %agg.result, i64 %x.coerce)
struct f2 func_f2(struct f2 x) { return x; }

// CHECK-LE: define { i64, i64 } @func_f3([2 x i64] %x.coerce)
// CHECK-BE: define void @func_f3(%struct.f3* noalias sret %agg.result, [2 x i64] %x.coerce)
struct f3 func_f3(struct f3 x) { return x; }

// CHECK-LE: define { i64, i64 } @func_f4([2 x i64] %x.coerce)
// CHECK-BE: define void @func_f4(%struct.f4* noalias sret %agg.result, [2 x i64] %x.coerce)
struct f4 func_f4(struct f4 x) { return x; }

// CHECK: define void @func_f5(%struct.f5* noalias sret %agg.result, [3 x i64] %x.coerce)
struct f5 func_f5(struct f5 x) { return x; }

// CHECK: define void @func_f6(%struct.f6* noalias sret %agg.result, [3 x i64] %x.coerce)
struct f6 func_f6(struct f6 x) { return x; }

// CHECK: define void @func_f7(%struct.f7* noalias sret %agg.result, [4 x i64] %x.coerce)
struct f7 func_f7(struct f7 x) { return x; }

// CHECK: define void @func_f8(%struct.f8* noalias sret %agg.result, [4 x i64] %x.coerce)
struct f8 func_f8(struct f8 x) { return x; }

// CHECK: define void @func_f9(%struct.f9* noalias sret %agg.result, [5 x i64] %x.coerce)
struct f9 func_f9(struct f9 x) { return x; }

// CHECK-LE: define i64 @func_fab(i64 %x.coerce)
// CHECK-BE: define void @func_fab(%struct.fab* noalias sret %agg.result, i64 %x.coerce)
struct fab func_fab(struct fab x) { return x; }

// CHECK-LE: define { i64, i64 } @func_fabc([2 x i64] %x.coerce)
// CHECK-BE: define void @func_fabc(%struct.fabc* noalias sret %agg.result, [2 x i64] %x.coerce)
struct fabc func_fabc(struct fabc x) { return x; }

// CHECK-LE: define { i64, i64 } @func_f2a2b([2 x i64] %x.coerce)
// CHECK-BE: define void @func_f2a2b(%struct.f2a2b* noalias sret %agg.result, [2 x i64] %x.coerce)
struct f2a2b func_f2a2b(struct f2a2b x) { return x; }

// CHECK-LABEL: @call_f1
// CHECK-BE: %[[TMP0:[^ ]+]] = alloca %struct.f1, align 4
// CHECK: %[[TMP:[^ ]+]] = load float, float* getelementptr inbounds (%struct.f1, %struct.f1* @global_f1, i32 0, i32 0, i32 0), align 4
// CHECK-LE: call i32 @func_f1(float inreg %[[TMP]])
// CHECK-BE: call void @func_f1(%struct.f1* sret %[[TMP0]], float inreg %[[TMP]])
struct f1 global_f1;
void call_f1(void) { global_f1 = func_f1(global_f1); }

// CHECK-LABEL: @call_f2
// CHECK-BE: %[[TMP0:[^ ]+]] = alloca %struct.f2, align 4
// CHECK: %[[TMP:[^ ]+]] = load i64, i64* bitcast (%struct.f2* @global_f2 to i64*), align 4
// CHECK-LE: call i64 @func_f2(i64 %[[TMP]])
// CHECK-BE: call void @func_f2(%struct.f2* sret %[[TMP0]], i64 %[[TMP]])
struct f2 global_f2;
void call_f2(void) { global_f2 = func_f2(global_f2); }

// CHECK-LABEL: @call_f3
// CHECK-BE: %[[TMP0:[^ ]+]] = alloca %struct.f3, align 4
// CHECK: %[[TMP1:[^ ]+]] = alloca [2 x i64]
// CHECK: %[[TMP2:[^ ]+]] = bitcast [2 x i64]* %[[TMP1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[TMP2]], i8* align 4 bitcast (%struct.f3* @global_f3 to i8*), i64 12, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [2 x i64], [2 x i64]* %[[TMP1]]
// CHECK-LE: call { i64, i64 } @func_f3([2 x i64] %[[TMP3]])
// CHECK-BE: call void @func_f3(%struct.f3* sret %[[TMP0]], [2 x i64] %[[TMP3]])
struct f3 global_f3;
void call_f3(void) { global_f3 = func_f3(global_f3); }

// CHECK-LABEL: @call_f4
// CHECK-BE: %[[TMP0:[^ ]+]] = alloca %struct.f4, align 4
// CHECK: %[[TMP:[^ ]+]] = load [2 x i64], [2 x i64]* bitcast (%struct.f4* @global_f4 to [2 x i64]*), align 4
// CHECK-LE: call { i64, i64 } @func_f4([2 x i64] %[[TMP]])
// CHECK-BE: call void @func_f4(%struct.f4* sret %[[TMP0]], [2 x i64] %[[TMP]])
struct f4 global_f4;
void call_f4(void) { global_f4 = func_f4(global_f4); }

// CHECK-LABEL: @call_f5
// CHECK: %[[TMP0:[^ ]+]] = alloca %struct.f5, align 4
// CHECK: %[[TMP1:[^ ]+]] = alloca [3 x i64]
// CHECK: %[[TMP2:[^ ]+]] = bitcast [3 x i64]* %[[TMP1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[TMP2]], i8* align 4 bitcast (%struct.f5* @global_f5 to i8*), i64 20, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [3 x i64], [3 x i64]* %[[TMP1]]
// CHECK: call void @func_f5(%struct.f5* sret %[[TMP0]], [3 x i64] %[[TMP3]])
struct f5 global_f5;
void call_f5(void) { global_f5 = func_f5(global_f5); }

// CHECK-LABEL: @call_f6
// CHECK: %[[TMP0:[^ ]+]] = alloca %struct.f6, align 4
// CHECK: %[[TMP:[^ ]+]] = load [3 x i64], [3 x i64]* bitcast (%struct.f6* @global_f6 to [3 x i64]*), align 4
// CHECK: call void @func_f6(%struct.f6* sret %[[TMP0]], [3 x i64] %[[TMP]])
struct f6 global_f6;
void call_f6(void) { global_f6 = func_f6(global_f6); }

// CHECK-LABEL: @call_f7
// CHECK: %[[TMP0:[^ ]+]] = alloca %struct.f7, align 4
// CHECK: %[[TMP1:[^ ]+]] = alloca [4 x i64], align 8
// CHECK: %[[TMP2:[^ ]+]] = bitcast [4 x i64]* %[[TMP1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[TMP2]], i8* align 4 bitcast (%struct.f7* @global_f7 to i8*), i64 28, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [4 x i64], [4 x i64]* %[[TMP1]], align 8
// CHECK: call void @func_f7(%struct.f7* sret %[[TMP0]], [4 x i64] %[[TMP3]])
struct f7 global_f7;
void call_f7(void) { global_f7 = func_f7(global_f7); }

// CHECK-LABEL: @call_f8
// CHECK: %[[TMP0:[^ ]+]] = alloca %struct.f8, align 4
// CHECK: %[[TMP:[^ ]+]] = load [4 x i64], [4 x i64]* bitcast (%struct.f8* @global_f8 to [4 x i64]*), align 4
// CHECK: call void @func_f8(%struct.f8* sret %[[TMP0]], [4 x i64] %[[TMP]])
struct f8 global_f8;
void call_f8(void) { global_f8 = func_f8(global_f8); }

// CHECK-LABEL: @call_f9
// CHECK: %[[TMP1:[^ ]+]] = alloca [5 x i64]
// CHECK: %[[TMP2:[^ ]+]] = bitcast [5 x i64]* %[[TMP1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[TMP2]], i8* align 4 bitcast (%struct.f9* @global_f9 to i8*), i64 36, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [5 x i64], [5 x i64]* %[[TMP1]]
// CHECK: call void @func_f9(%struct.f9* sret %{{[^ ]+}}, [5 x i64] %[[TMP3]])
struct f9 global_f9;
void call_f9(void) { global_f9 = func_f9(global_f9); }

// CHECK-LABEL: @call_fab
// CHECK: %[[TMP0:[^ ]+]] = alloca %struct.fab, align 4
// CHECK: %[[TMP:[^ ]+]] = load i64, i64* bitcast (%struct.fab* @global_fab to i64*), align 4
// CHECK-LE: %call = call i64 @func_fab(i64 %[[TMP]])
// CHECK-BE: call void @func_fab(%struct.fab* sret %[[TMP0]], i64 %[[TMP]])
struct fab global_fab;
void call_fab(void) { global_fab = func_fab(global_fab); }

// CHECK-LABEL: @call_fabc
// CHECK-BE: %[[TMPX:[^ ]+]] = alloca %struct.fabc, align 4
// CHECK: %[[TMP0:[^ ]+]] = alloca [2 x i64], align 8
// CHECK: %[[TMP2:[^ ]+]] = bitcast [2 x i64]* %[[TMP0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[TMP2]], i8* align 4 bitcast (%struct.fabc* @global_fabc to i8*), i64 12, i1 false)
// CHECK: %[[TMP3:[^ ]+]] = load [2 x i64], [2 x i64]* %[[TMP0]], align 8
// CHECK-LE: %call = call { i64, i64 } @func_fabc([2 x i64] %[[TMP3]])
// CHECK-BE: call void @func_fabc(%struct.fabc* sret %[[TMPX]], [2 x i64] %[[TMP3]])
struct fabc global_fabc;
void call_fabc(void) { global_fabc = func_fabc(global_fabc); }

