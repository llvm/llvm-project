// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -o - %s | FileCheck %s

// Test homogeneous fp128 aggregate passing and returning.

struct fp1 { __float128 f[1]; };
struct fp2 { __float128 f[2]; };
struct fp3 { __float128 f[3]; };
struct fp4 { __float128 f[4]; };
struct fp5 { __float128 f[5]; };
struct fp6 { __float128 f[6]; };
struct fp7 { __float128 f[7]; };
struct fp8 { __float128 f[8]; };
struct fp9 { __float128 f[9]; };

struct fpab { __float128 a; __float128 b; };
struct fpabc { __float128 a; __float128 b; __float128 c; };

struct fp2a2b { __float128 a[2]; __float128 b[2]; };

// CHECK: define [1 x fp128] @func_f1(fp128 inreg %x.coerce)
struct fp1 func_f1(struct fp1 x) { return x; }

// CHECK: define [2 x fp128] @func_f2([2 x fp128] %x.coerce)
struct fp2 func_f2(struct fp2 x) { return x; }

// CHECK: define [3 x fp128] @func_f3([3 x fp128] %x.coerce)
struct fp3 func_f3(struct fp3 x) { return x; }

// CHECK: define [4 x fp128] @func_f4([4 x fp128] %x.coerce)
struct fp4 func_f4(struct fp4 x) { return x; }

// CHECK: define [5 x fp128] @func_f5([5 x fp128] %x.coerce)
struct fp5 func_f5(struct fp5 x) { return x; }

// CHECK: define [6 x fp128] @func_f6([6 x fp128] %x.coerce)
struct fp6 func_f6(struct fp6 x) { return x; }

// CHECK: define [7 x fp128] @func_f7([7 x fp128] %x.coerce)
struct fp7 func_f7(struct fp7 x) { return x; }

// CHECK: define [8 x fp128] @func_f8([8 x fp128] %x.coerce)
struct fp8 func_f8(struct fp8 x) { return x; }

// CHECK: define void @func_f9(%struct.fp9* noalias sret(%struct.fp9) align 16 %agg.result, %struct.fp9* byval(%struct.fp9) align 16 %x)
struct fp9 func_f9(struct fp9 x) { return x; }

// CHECK: define [2 x fp128] @func_fab([2 x fp128] %x.coerce)
struct fpab func_fab(struct fpab x) { return x; }

// CHECK: define [3 x fp128] @func_fabc([3 x fp128] %x.coerce)
struct fpabc func_fabc(struct fpabc x) { return x; }

// CHECK: define [4 x fp128] @func_f2a2b([4 x fp128] %x.coerce)
struct fp2a2b func_f2a2b(struct fp2a2b x) { return x; }

// CHECK-LABEL: @call_fp1
// CHECK: %[[TMP:[^ ]+]] = load fp128, fp128* getelementptr inbounds (%struct.fp1, %struct.fp1* @global_f1, i32 0, i32 0, i32 0), align 16
// CHECK: call [1 x fp128] @func_f1(fp128 inreg %[[TMP]])
struct fp1 global_f1;
void call_fp1(void) { global_f1 = func_f1(global_f1); }

// CHECK-LABEL: @call_fp2
// CHECK: %[[TMP:[^ ]+]] = load [2 x fp128], [2 x fp128]* getelementptr inbounds (%struct.fp2, %struct.fp2* @global_f2, i32 0, i32 0), align 16
// CHECK: call [2 x fp128] @func_f2([2 x fp128] %[[TMP]])
struct fp2 global_f2;
void call_fp2(void) { global_f2 = func_f2(global_f2); }

// CHECK-LABEL: @call_fp3
// CHECK: %[[TMP:[^ ]+]] = load [3 x fp128], [3 x fp128]* getelementptr inbounds (%struct.fp3, %struct.fp3* @global_f3, i32 0, i32 0), align 16
// CHECK: call [3 x fp128] @func_f3([3 x fp128] %[[TMP]])
struct fp3 global_f3;
void call_fp3(void) { global_f3 = func_f3(global_f3); }

// CHECK-LABEL: @call_fp4
// CHECK: %[[TMP:[^ ]+]] = load [4 x fp128], [4 x fp128]* getelementptr inbounds (%struct.fp4, %struct.fp4* @global_f4, i32 0, i32 0), align 16
// CHECK: call [4 x fp128] @func_f4([4 x fp128] %[[TMP]])
struct fp4 global_f4;
void call_fp4(void) { global_f4 = func_f4(global_f4); }

// CHECK-LABEL: @call_fp5
// CHECK: %[[TMP:[^ ]+]] = load [5 x fp128], [5 x fp128]* getelementptr inbounds (%struct.fp5, %struct.fp5* @global_f5, i32 0, i32 0), align 16
// CHECK: call [5 x fp128] @func_f5([5 x fp128] %[[TMP]])
struct fp5 global_f5;
void call_fp5(void) { global_f5 = func_f5(global_f5); }

// CHECK-LABEL: @call_fp6
// CHECK: %[[TMP:[^ ]+]] = load [6 x fp128], [6 x fp128]* getelementptr inbounds (%struct.fp6, %struct.fp6* @global_f6, i32 0, i32 0), align 16
// CHECK: call [6 x fp128] @func_f6([6 x fp128] %[[TMP]])
struct fp6 global_f6;
void call_fp6(void) { global_f6 = func_f6(global_f6); }

// CHECK-LABEL: @call_fp7
// CHECK: %[[TMP:[^ ]+]] = load [7 x fp128], [7 x fp128]* getelementptr inbounds (%struct.fp7, %struct.fp7* @global_f7, i32 0, i32 0), align 16
// CHECK: call [7 x fp128] @func_f7([7 x fp128] %[[TMP]])
struct fp7 global_f7;
void call_fp7(void) { global_f7 = func_f7(global_f7); }

// CHECK-LABEL: @call_fp8
// CHECK: %[[TMP:[^ ]+]] = load [8 x fp128], [8 x fp128]* getelementptr inbounds (%struct.fp8, %struct.fp8* @global_f8, i32 0, i32 0), align 16
// CHECK: call [8 x fp128] @func_f8([8 x fp128] %[[TMP]])
struct fp8 global_f8;
void call_fp8(void) { global_f8 = func_f8(global_f8); }

// CHECK-LABEL: @call_fp9
// CHECK: %[[TMP1:[^ ]+]] = alloca %struct.fp9, align 16
// CHECK: call void @func_f9(%struct.fp9* sret(%struct.fp9) align 16 %[[TMP2:[^ ]+]], %struct.fp9* byval(%struct.fp9) align 16 @global_f9
// CHECK: %[[TMP3:[^ ]+]] = bitcast %struct.fp9* %[[TMP2]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 bitcast (%struct.fp9* @global_f9 to i8*), i8* align 16 %[[TMP3]], i64 144, i1 false
// CHECK: ret void
struct fp9 global_f9;
void call_fp9(void) { global_f9 = func_f9(global_f9); }

// CHECK-LABEL: @call_fpab
// CHECK: %[[TMP:[^ ]+]] = load [2 x fp128], [2 x fp128]* bitcast (%struct.fpab* @global_fab to [2 x fp128]*)
// CHECK: call [2 x fp128] @func_fab([2 x fp128] %[[TMP]])
struct fpab global_fab;
void call_fpab(void) { global_fab = func_fab(global_fab); }

// CHECK-LABEL: @call_fpabc
// CHECK: %[[TMP:[^ ]+]] = load [3 x fp128], [3 x fp128]* bitcast (%struct.fpabc* @global_fabc to [3 x fp128]*)
// CHECK: call [3 x fp128] @func_fabc([3 x fp128] %[[TMP]])
struct fpabc global_fabc;
void call_fpabc(void) { global_fabc = func_fabc(global_fabc); }
