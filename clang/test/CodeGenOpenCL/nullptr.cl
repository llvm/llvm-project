// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple spir64 -emit-llvm -o - -Wno-void-pointer-to-int-cast -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast | FileCheck %s --check-prefixes=CHECK,SPIR64
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -O0 -cl-std=CL2.0 -include opencl-c.h -triple spir64 -emit-llvm -o - -Wno-void-pointer-to-int-cast -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast | FileCheck --check-prefixes=CHECK-NOOPT,SPIR64-NOOPT %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,AMDGCN
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -O0 -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -emit-llvm -o - | FileCheck --check-prefixes=CHECK-NOOPT,AMDGCN-NOOPT %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple amdgcn---opencl -emit-llvm -o - | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -fcommon -emit-llvm -o - | FileCheck %s --check-prefix=AMDGCN-COMMON

typedef struct {
  private char *p1;
  local char *p2;
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy1;

typedef struct {
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy2;

// Test 0 as initializer.

// SPIR64: @private_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// AMDGCN: @private_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
private char *private_p = 0;

// SPIR64: @local_p = local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// AMDGCN: @local_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
local char *local_p = 0;

// SPIR64: @global_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), align 8
// AMDGCN: @global_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) null, align 8
global char *global_p = 0;

// SPIR64: @constant_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(2) null, align 8
// AMDGCN: @constant_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
constant char *constant_p = 0;

// SPIR64: @generic_p ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
// AMDGCN: @generic_p ={{.*}} local_unnamed_addr addrspace(1) global ptr null, align 8
generic char *generic_p = 0;

// Test NULL as initializer.

// SPIR64: @private_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// AMDGCN: @private_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
private char *private_p_NULL = NULL;

// SPIR64: @local_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// AMDGCN: @local_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
local char *local_p_NULL = NULL;

// SPIR64: @global_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), align 8
// AMDGCN: @global_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) null, align 8
global char *global_p_NULL = NULL;

// SPIR64: @constant_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(2) null, align 8
// AMDGCN: @constant_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
constant char *constant_p_NULL = NULL;

// SPIR64: @generic_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
// AMDGCN: @generic_p_NULL ={{.*}} local_unnamed_addr addrspace(1) global ptr null, align 8
generic char *generic_p_NULL = NULL;

// Test constant folding of null pointer.
// A null pointer should be folded to a null pointer in the target address space.

// SPIR64: @fold_generic ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
// AMDGCN: @fold_generic ={{.*}} local_unnamed_addr addrspace(1) global ptr null, align 8
generic int *fold_generic = (global int*)(generic float*)(private char*)0;

// SPIR64: @fold_priv ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// AMDGCN: @fold_priv ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr addrspace(1) null to ptr addrspace(5)), align 4
private short *fold_priv = (private short*)(generic int*)(global void*)0;

// SPIR64: @fold_priv_arith ={{.*}} local_unnamed_addr addrspace(1) global ptr inttoptr (i64 10 to ptr), align 8
// AMDGCN: @fold_priv_arith ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) inttoptr (i32 9 to ptr addrspace(5)), align 4
private char *fold_priv_arith = (private char*)0 + 10;

// SPIR64: @fold_local_arith ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) inttoptr (i64 10 to ptr addrspace(3)), align 8
// AMDGCN: @fold_local_arith ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) inttoptr (i32 9 to ptr addrspace(3)), align 4
local char *fold_local_arith = (local char*)0 + 10;

// SPIR64: @fold_int ={{.*}} local_unnamed_addr addrspace(1) global i32 14, align 4
// AMDGCN: @fold_int ={{.*}} local_unnamed_addr addrspace(1) global i32 13, align 4
int fold_int = (int)(private void*)(generic char*)(global int*)0 + 14;

// SPIR64: @fold_int2 ={{.*}} local_unnamed_addr addrspace(1) global i32 13, align 4
// AMDGCN: @fold_int2 ={{.*}} local_unnamed_addr addrspace(1) global i32 12, align 4
int fold_int2 = (int) ((private void*)0 + 13);

// SPIR64: @fold_int3 ={{.*}} local_unnamed_addr addrspace(1) global i32 0, align 4
// AMDGCN: @fold_int3 ={{.*}} local_unnamed_addr addrspace(1) global i32 -1, align 4
int fold_int3 = (int) ((private int*)0);

// SPIR64: @fold_int4 ={{.*}} local_unnamed_addr addrspace(1) global i32 8, align 4
// AMDGCN: @fold_int4 ={{.*}} local_unnamed_addr addrspace(1) global i32 7, align 4
int fold_int4 = (int) &((private int*)0)[2];

// SPIR64: @fold_int5 ={{.*}} local_unnamed_addr addrspace(1) global i32 8, align 4
// AMDGCN: @fold_int5 ={{.*}} local_unnamed_addr addrspace(1) global i32 3, align 4
int fold_int5 = (int) &((private StructTy1*)0)->p2;

// SPIR64: @fold_int_local ={{.*}} local_unnamed_addr addrspace(1) global i32 14, align 4
// AMDGCN: @fold_int_local = local_unnamed_addr addrspace(1) global i32 13, align 4
int fold_int_local = (int)(local void*)(generic char*)(global int*)0 + 14;

// SPIR64: @fold_int2_local ={{.*}} local_unnamed_addr addrspace(1) global i32 13, align 4
// AMDGCN: @fold_int2_local ={{.*}} local_unnamed_addr addrspace(1) global i32 12, align 4
int fold_int2_local = (int) ((local void*)0 + 13);

// SPIR64: @fold_int3_local ={{.*}} local_unnamed_addr addrspace(1) global i32 0, align 4
// AMDGCN: @fold_int3_local ={{.*}} local_unnamed_addr addrspace(1) global i32 -1, align 4
int fold_int3_local = (int) ((local int*)0);

// SPIR64: @fold_int4_local ={{.*}} local_unnamed_addr addrspace(1) global i32 8, align 4
// AMDGCN: @fold_int4_local ={{.*}} local_unnamed_addr addrspace(1) global i32 7, align 4
int fold_int4_local = (int) &((local int*)0)[2];

// SPIR64: @fold_int5_local ={{.*}} local_unnamed_addr addrspace(1) global i32 8, align 4
// AMDGCN: @fold_int5_local ={{.*}} local_unnamed_addr addrspace(1) global i32 3, align 4
int fold_int5_local = (int) &((local StructTy1*)0)->p2;


// Test static variable initialization.

// SPIR64-NOOPT: @test_static_var_private.sp1 = internal addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// SPIR64-NOOPT: @test_static_var_private.sp2 = internal addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// SPIR64-NOOPT: @test_static_var_private.sp3 = internal addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// SPIR64-NOOPT: @test_static_var_private.sp4 = internal addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// SPIR64-NOOPT: @test_static_var_private.sp5 = internal addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// SPIR64-NOOPT: @test_static_var_private.SS1 = internal addrspace(1) global %struct.StructTy1 zeroinitializer, align 8
// AMDGCN-NOOPT: @test_static_var_private.sp1 = internal addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
// AMDGCN-NOOPT: @test_static_var_private.sp2 = internal addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
// AMDGCN-NOOPT: @test_static_var_private.sp3 = internal addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
// AMDGCN-NOOPT: @test_static_var_private.sp4 = internal addrspace(1) global ptr addrspace(5) null, align 4
// AMDGCN-NOOPT: @test_static_var_private.sp5 = internal addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
// AMDGCN-NOOPT: @test_static_var_private.SS1 = internal addrspace(1) global %struct.StructTy1 { ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(4) null, ptr addrspace(1) null, ptr null }, align 8
// CHECK-NOOPT: @test_static_var_private.SS2 = internal addrspace(1) global %struct.StructTy2 zeroinitializer, align 8

void test_static_var_private(void) {
  static private char *sp1 = 0;
  static private char *sp2 = NULL;
  static private char *sp3;
  static private char *sp4 = (private char*)((void)0, 0);
  const int x = 0;
  static private char *sp5 = (private char*)x;
  static StructTy1 SS1;
  static StructTy2 SS2;
}

// SPIR64-NOOPT: @test_static_var_local.sp1 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// SPIR64-NOOPT: @test_static_var_local.sp2 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// SPIR64-NOOPT: @test_static_var_local.sp3 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// SPIR64-NOOPT: @test_static_var_local.sp4 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// SPIR64-NOOPT: @test_static_var_local.sp5 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// SPIR64-NOOPT: @test_static_var_local.SS1 = internal addrspace(1) global %struct.StructTy1 zeroinitializer, align 8
// AMDGCN-NOOPT: @test_static_var_local.sp1 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
// AMDGCN-NOOPT: @test_static_var_local.sp2 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
// AMDGCN-NOOPT: @test_static_var_local.sp3 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
// AMDGCN-NOOPT: @test_static_var_local.sp4 = internal addrspace(1) global ptr addrspace(3) null, align 4
// AMDGCN-NOOPT: @test_static_var_local.sp5 = internal addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
// AMDGCN-NOOPT: @test_static_var_local.SS1 = internal addrspace(1) global %struct.StructTy1 { ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(4) null, ptr addrspace(1) null, ptr null }, align 8
// CHECK-NOOPT: @test_static_var_local.SS2 = internal addrspace(1) global %struct.StructTy2 zeroinitializer, align 8
void test_static_var_local(void) {
  static local char *sp1 = 0;
  static local char *sp2 = NULL;
  static local char *sp3;
  static local char *sp4 = (local char*)((void)0, 0);
  const int x = 0;
  static local char *sp5 = (local char*)x;
  static StructTy1 SS1;
  static StructTy2 SS2;
}

// Test function-scope variable initialization.
// CHECK-NOOPT-LABEL: @test_func_scope_var_private(
// SPIR64-NOOPT: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr %sp1{{.*}}, align 8
// SPIR64-NOOPT: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr %sp2{{.*}}, align 8
// SPIR64-NOOPT: store ptr null, ptr %sp3{{.*}}, align 8
// SPIR64-NOOPT: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr %sp4{{.*}}, align 8
// SPIR64-NOOPT: call void @llvm.memset.p0.i64(ptr align 8 %SS1{{.*}}, i8 0, i64 40, i1 false)
// SPIR64-NOOPT: call void @llvm.memcpy.p0.p2.i64(ptr align 8 %SS2{{.*}}, ptr addrspace(2) align 8 @__const.test_func_scope_var_private.SS2, i64 24, i1 false)
// AMDGCN-NOOPT: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) %sp1{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) %sp2{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(5) null, ptr addrspace(5) %sp3{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) %sp4{{.*}}, align 4
// AMDGCN-NOOPT: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %SS1{{.*}}, ptr addrspace(4) align 8 @__const.test_func_scope_var_private.SS1, i64 32, i1 false)
// AMDGCN-NOOPT: call void @llvm.memset.p5.i64(ptr addrspace(5) align 8 %SS2{{.*}}, i8 0, i64 24, i1 false)
void test_func_scope_var_private(void) {
  private char *sp1 = 0;
  private char *sp2 = NULL;
  private char *sp3 = (private char*)((void)0, 0);
  const int x = 0;
  private char *sp4 = (private char*)x;
  StructTy1 SS1 = {0, 0, 0, 0, 0};
  StructTy2 SS2 = {0, 0, 0};
}

// Test function-scope variable initialization.
// CHECK-NOOPT-LABEL: @test_func_scope_var_local(
// SPIR64-NOOPT: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr %sp1{{.*}}, align 8
// SPIR64-NOOPT: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr %sp2{{.*}}, align 8
// SPIR64-NOOPT: store ptr addrspace(3) null, ptr %sp3{{.*}}, align 8
// SPIR64-NOOPT: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr %sp4{{.*}}, align 8
// SPIR64-NOOPT: call void @llvm.memset.p0.i64(ptr align 8 %SS1{{.*}}, i8 0, i64 40, i1 false)
// SPIR64-NOOPT: call void @llvm.memcpy.p0.p2.i64(ptr align 8 %SS2{{.*}}, ptr addrspace(2) align 8 @__const.test_func_scope_var_local.SS2, i64 24, i1 false)
// AMDGCN-NOOPT: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(5) %sp1{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(5) %sp2{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(3) null, ptr addrspace(5) %sp3{{.*}}, align 4
// AMDGCN-NOOPT: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(5) %sp4{{.*}}, align 4
// AMDGCN-NOOPT: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 8 %SS1{{.*}}, ptr addrspace(4) align 8 @__const.test_func_scope_var_local.SS1, i64 32, i1 false)
// AMDGCN-NOOPT: call void @llvm.memset.p5.i64(ptr addrspace(5) align 8 %SS2{{.*}}, i8 0, i64 24, i1 false)
void test_func_scope_var_local(void) {
  local char *sp1 = 0;
  local char *sp2 = NULL;
  local char *sp3 = (local char*)((void)0, 0);
  const int x = 0;
  local char *sp4 = (local char*)x;
  StructTy1 SS1 = {0, 0, 0, 0, 0};
  StructTy2 SS2 = {0, 0, 0};
}


// Test default initialization of pointers.

// Tentative definition of global variables with non-zero initializer
// cannot have common linkage since common linkage requires zero initialization
// and does not have explicit section.

// SPIR64: @p1 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspacecast (ptr addrspace(4) null to ptr), align 8
// AMDGCN: @p1 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
// AMDGCN-COMMON: @p1 = weak local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4
private char *p1;

// SPIR64: @p2 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), align 8
// AMDGCN: @p2 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
// AMDGCN-COMMON: @p2 = weak local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
local char *p2;

// SPIR64: @p3 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(2) null, align 8
// AMDGCN: @p3 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
// AMDGCN-COMMON: @p3 = common local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
constant char *p3;

// SPIR64: @p4 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), align 8
// AMDGCN: @p4 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(1) null, align 8
// AMDGCN-COMMON: @p4 = common local_unnamed_addr addrspace(1) global ptr addrspace(1) null, align 8
global char *p4;

// SPIR64: @p5 ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(4) null, align 8
// AMDGCN: @p5 ={{.*}} local_unnamed_addr addrspace(1) global ptr null, align 8
// AMDGCN-COMMON: @p5 = common local_unnamed_addr addrspace(1) global ptr null, align 8
generic char *p5;

// Test default initialization of structure.

// SPIR64: @S1 ={{.*}} local_unnamed_addr addrspace(1) global %struct.StructTy1 zeroinitializer, align 8
// AMDGCN: @S1 ={{.*}} local_unnamed_addr addrspace(1) global %struct.StructTy1 { ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(4) null, ptr addrspace(1) null, ptr null }, align 8
StructTy1 S1;

// CHECK: @S2 ={{.*}} local_unnamed_addr addrspace(1) global %struct.StructTy2 zeroinitializer, align 8
StructTy2 S2;

// Test default initialization of array.
// SPIR64: @A1 ={{.*}} local_unnamed_addr addrspace(1) global [2 x %struct.StructTy1] zeroinitializer, align 8
// AMDGCN: @A1 ={{.*}} local_unnamed_addr addrspace(1) global [2 x %struct.StructTy1] [%struct.StructTy1 { ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(4) null, ptr addrspace(1) null, ptr null }, %struct.StructTy1 { ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(4) null, ptr addrspace(1) null, ptr null }], align 8
StructTy1 A1[2];

// CHECK: @A2 ={{.*}} local_unnamed_addr addrspace(1) global [2 x %struct.StructTy2] zeroinitializer, align 8
StructTy2 A2[2];

// Test comparison with 0.

// CHECK-LABEL: cmp_private
// SPIR64: icmp eq ptr %p, addrspacecast (ptr addrspace(4) null to ptr)
// AMDGCN: icmp eq ptr addrspace(5) %p, addrspacecast (ptr null to ptr addrspace(5))
void cmp_private(private char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_local
// SPIR64: icmp eq ptr addrspace(3) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: icmp eq ptr addrspace(3) %p, addrspacecast (ptr null to ptr addrspace(3))
void cmp_local(local char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_global
// SPIR64: icmp eq ptr addrspace(1) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(1))
// AMDGCN: icmp eq ptr addrspace(1) %p, null
void cmp_global(global char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_constant
// SPIR64: icmp eq ptr addrspace(2) %p, null
// AMDGCN: icmp eq ptr addrspace(4) %p, null
char cmp_constant(constant char* p) {
  if (p != 0)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cmp_generic
// SPIR64: icmp eq ptr addrspace(4) %p, null
// AMDGCN: icmp eq ptr %p, null
void cmp_generic(generic char* p) {
  if (p != 0)
    *p = 0;
}

// Test comparison with NULL.

// CHECK-LABEL: cmp_NULL_private
// SPIR64: icmp eq ptr %p, addrspacecast (ptr addrspace(4) null to ptr)
// AMDGCN: icmp eq ptr addrspace(5) %p, addrspacecast (ptr null to ptr addrspace(5))
void cmp_NULL_private(private char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_local
// SPIR64: icmp eq ptr addrspace(3) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: icmp eq ptr addrspace(3) %p, addrspacecast (ptr null to ptr addrspace(3))
void cmp_NULL_local(local char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_global
// SPIR64: icmp eq ptr addrspace(1) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(1))
// AMDGCN: icmp eq ptr addrspace(1) %p, null
void cmp_NULL_global(global char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_constant
// SPIR64: icmp eq ptr addrspace(2) %p, null
// AMDGCN: icmp eq ptr addrspace(4) %p, null
char cmp_NULL_constant(constant char* p) {
  if (p != NULL)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cmp_NULL_generic
// SPIR64: icmp eq ptr addrspace(4) %p, null
// AMDGCN: icmp eq ptr %p, null
void cmp_NULL_generic(generic char* p) {
  if (p != NULL)
    *p = 0;
}

// Test storage 0 as null pointer.
// CHECK-LABEL: test_storage_null_pointer
// SPIR64: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr addrspace(4) %arg_private
// SPIR64: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(4) %arg_local
// SPIR64: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(4) %arg_global
// SPIR64: store ptr addrspace(2) null, ptr addrspace(4) %arg_constant
// SPIR64: store ptr addrspace(4) null, ptr addrspace(4) %arg_generic
// AMDGCN: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr %arg_private
// AMDGCN: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr %arg_local
// AMDGCN: store ptr addrspace(1) null, ptr %arg_global
// AMDGCN: store ptr addrspace(4) null, ptr %arg_constant
// AMDGCN: store ptr null, ptr %arg_generic
void test_storage_null_pointer(private char** arg_private,
                               local char** arg_local,
                               global char** arg_global,
                               constant char** arg_constant,
                               generic char** arg_generic) {
   *arg_private = 0;
   *arg_local = 0;
   *arg_global = 0;
   *arg_constant = 0;
   *arg_generic = 0;
}

// Test storage NULL as null pointer.
// CHECK-LABEL: test_storage_null_pointer_NULL
// SPIR64: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr addrspace(4) %arg_private
// SPIR64: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(4) %arg_local
// SPIR64: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(4) %arg_global
// SPIR64: store ptr addrspace(2) null, ptr addrspace(4) %arg_constant
// SPIR64: store ptr addrspace(4) null, ptr addrspace(4) %arg_generic
// AMDGCN: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr %arg_private
// AMDGCN: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr %arg_local
// AMDGCN: store ptr addrspace(1) null, ptr %arg_global
// AMDGCN: store ptr addrspace(4) null, ptr %arg_constant
// AMDGCN: store ptr null, ptr %arg_generic
void test_storage_null_pointer_NULL(private char** arg_private,
                                    local char** arg_local,
                                    global char** arg_global,
                                    constant char** arg_constant,
                                    generic char** arg_generic) {
   *arg_private = NULL;
   *arg_local = NULL;
   *arg_global = NULL;
   *arg_constant = NULL;
   *arg_generic = NULL;
}

// Test pass null pointer to function as argument.
void test_pass_null_pointer_arg_calee(private char* arg_private,
                                      local char* arg_local,
                                      global char* arg_global,
                                      constant char* arg_constant,
                                      generic char* arg_generic);

// CHECK-LABEL: test_pass_null_pointer_arg
// SPIR64: call spir_func void @test_pass_null_pointer_arg_calee(ptr addrspacecast (ptr addrspace(4) null to ptr), ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(2) null, ptr addrspace(4) null)
// SPIR64: call spir_func void @test_pass_null_pointer_arg_calee(ptr addrspacecast (ptr addrspace(4) null to ptr), ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(2) null, ptr addrspace(4) null)
// AMDGCN: call void @test_pass_null_pointer_arg_calee(ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(1) null, ptr addrspace(4) null, ptr null)
// AMDGCN: call void @test_pass_null_pointer_arg_calee(ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(1) null, ptr addrspace(4) null, ptr null)
void test_pass_null_pointer_arg(void) {
  test_pass_null_pointer_arg_calee(0, 0, 0, 0, 0);
  test_pass_null_pointer_arg_calee(NULL, NULL, NULL, NULL, NULL);
}

// Test cast null pointer to size_t.
void test_cast_null_pointer_to_sizet_calee(size_t arg_private,
                                           size_t arg_local,
                                           size_t arg_global,
                                           size_t arg_constant,
                                           size_t arg_generic);

// CHECK-LABEL: test_cast_null_pointer_to_sizet
// SPIR64: call spir_func void @test_cast_null_pointer_to_sizet_calee(i64 ptrtoint (ptr addrspacecast (ptr addrspace(4) null to ptr) to i64), i64 ptrtoint (ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)) to i64), i64 ptrtoint (ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)) to i64), i64 0, i64 0)
// SPIR64: call spir_func void @test_cast_null_pointer_to_sizet_calee(i64 ptrtoint (ptr addrspacecast (ptr addrspace(4) null to ptr) to i64), i64 ptrtoint (ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)) to i64), i64 ptrtoint (ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)) to i64), i64 0, i64 0)
// AMDGCN: call void @test_cast_null_pointer_to_sizet_calee(i64 ptrtoint (ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)) to i64), i64 ptrtoint (ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)) to i64), i64 0, i64 0, i64 0)
// AMDGCN: call void @test_cast_null_pointer_to_sizet_calee(i64 ptrtoint (ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)) to i64), i64 ptrtoint (ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)) to i64), i64 0, i64 0, i64 0)
void test_cast_null_pointer_to_sizet(void) {
  test_cast_null_pointer_to_sizet_calee((size_t)((private char*)0),
                                        (size_t)((local char*)0),
                                        (size_t)((global char*)0),
                                        (size_t)((constant char*)0),
                                        (size_t)((generic char*)0));
  test_cast_null_pointer_to_sizet_calee((size_t)((private char*)NULL),
                                        (size_t)((local char*)NULL),
                                        (size_t)((global char*)NULL),
                                        (size_t)((constant char*)0), // NULL cannot be casted to constant pointer since it is defined as a generic pointer
                                        (size_t)((generic char*)NULL));
}

// Test comparison between null pointers.
#define TEST_EQ00(addr1, addr2) int test_eq00_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)0; }
#define TEST_EQ0N(addr1, addr2) int test_eq0N_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)NULL; }
#define TEST_EQN0(addr1, addr2) int test_eqN0_##addr1##_##addr2(void) { return (addr1 char*)NULL == (addr2 char*)0; }
#define TEST_EQNN(addr1, addr2) int test_eqNN_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)NULL; }
#define TEST_NE00(addr1, addr2) int test_ne00_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)0; }
#define TEST_NE0N(addr1, addr2) int test_ne0N_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)NULL; }
#define TEST_NEN0(addr1, addr2) int test_neN0_##addr1##_##addr2(void) { return (addr1 char*)NULL != (addr2 char*)0; }
#define TEST_NENN(addr1, addr2) int test_neNN_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)NULL; }
#define TEST(addr1, addr2) \
        TEST_EQ00(addr1, addr2) \
        TEST_EQ0N(addr1, addr2) \
        TEST_EQN0(addr1, addr2) \
        TEST_EQNN(addr1, addr2) \
        TEST_NE00(addr1, addr2) \
        TEST_NE0N(addr1, addr2) \
        TEST_NEN0(addr1, addr2) \
        TEST_NENN(addr1, addr2)

// CHECK-LABEL: test_eq00_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_private
// CHECK: ret i32 0
TEST(generic, private)

// CHECK-LABEL: test_eq00_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_local
// CHECK: ret i32 0
TEST(generic, local)

// CHECK-LABEL: test_eq00_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_global
// CHECK: ret i32 0
TEST(generic, global)

// CHECK-LABEL: test_eq00_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_generic
// CHECK: ret i32 0
TEST(generic, generic)

// CHECK-LABEL: test_eq00_constant_constant
// CHECK: ret i32 1
TEST_EQ00(constant, constant)

// Test cast to bool.

// CHECK-LABEL: cast_bool_private
// SPIR64: icmp eq ptr %p, addrspacecast (ptr addrspace(4) null to ptr)
// AMDGCN: icmp eq ptr addrspace(5) %p, addrspacecast (ptr null to ptr addrspace(5))
void cast_bool_private(private char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_local
// SPIR64: icmp eq ptr addrspace(3) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: icmp eq ptr addrspace(3) %p, addrspacecast (ptr null to ptr addrspace(3))
void cast_bool_local(local char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_global
// SPIR64: icmp eq ptr addrspace(1) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(1))
// AMDGCN: icmp eq ptr addrspace(1) %p, null
void cast_bool_global(global char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_constant
// SPIR64: icmp eq ptr addrspace(2) %p, null
// AMDGCN: icmp eq ptr addrspace(4) %p, null
char cast_bool_constant(constant char* p) {
  if (p)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cast_bool_generic
// SPIR64: icmp eq ptr addrspace(4) %p, null
// AMDGCN: icmp eq ptr %p, null
void cast_bool_generic(generic char* p) {
  if (p)
    *p = 0;
}

// Test initialize a struct using memset.
// For large structures which is mostly zero, clang generats llvm.memset for
// the zero part and store for non-zero members.
typedef struct {
  long a, b, c, d;
  private char *p;
} StructTy3;

// CHECK-LABEL: test_memset_private
// SPIR64: call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %ptr, i8 0, i64 32, i1 false)
// SPIR64: [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %ptr, i64 32
// SPIR64: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr [[GEP]], align 8
// AMDGCN: call void @llvm.memset.p5.i64(ptr addrspace(5) noundef align 8 {{.*}}, i8 0, i64 32, i1 false)
// AMDGCN: [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(5) %ptr, i32 32
// AMDGCN: store ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) [[GEP]]
// AMDGCN: [[GEP1:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(5) {{.*}}, i32 36
// AMDGCN: store i32 0, ptr addrspace(5) [[GEP1]], align 4
void test_memset_private(private StructTy3 *ptr) {
  StructTy3 S3 = {0, 0, 0, 0, 0};
  *ptr = S3;
}

// Test casting literal 0 to pointer.
// A 0 literal casted to pointer should become a null pointer.

// CHECK-LABEL: test_cast_0_to_local_ptr
// SPIR64: ret ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: ret ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))
local int* test_cast_0_to_local_ptr(void) {
  return (local int*)0;
}

// CHECK-LABEL: test_cast_0_to_private_ptr
// SPIR64: ptr addrspacecast (ptr addrspace(4) null to ptr)
// AMDGCN: ret ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5))
private int* test_cast_0_to_private_ptr(void) {
  return (private int*)0;
}

// Test casting non-literal integer with 0 value to pointer.
// A non-literal integer expression with 0 value is casted to a pointer with
// zero value.

// CHECK-LABEL: test_cast_int_to_ptr1_private
// SPIR64: ret ptr null
// AMDGCN: ret ptr addrspace(5) null
private int* test_cast_int_to_ptr1_private(void) {
  return (private int*)((void)0, 0);
}

// CHECK-LABEL: test_cast_int_to_ptr1_local
// CHECK: ret ptr addrspace(3) null
local int* test_cast_int_to_ptr1_local(void) {
  return (local int*)((void)0, 0);
}

// CHECK-LABEL: test_cast_int_to_ptr2
// SPIR64: ret ptr null
// AMDGCN: ret ptr addrspace(5) null
private int* test_cast_int_to_ptr2(void) {
  int x = 0;
  return (private int*)x;
}

// Test logical operations.
// CHECK-LABEL: test_not_nullptr
// CHECK: ret i32 1
int test_not_nullptr(void) {
  return !(private char*)NULL;
}

// CHECK-LABEL: test_and_nullptr
// CHECK: ret i32 0
int test_and_nullptr(int a) {
  return a && ((private char*)NULL);
}

// CHECK-LABEL: test_not_private_ptr
// SPIR64: %[[lnot:.*]] = icmp eq ptr %p, addrspacecast (ptr addrspace(4) null to ptr)
// AMDGCN: %[[lnot:.*]] = icmp eq ptr addrspace(5) %p, addrspacecast (ptr null to ptr addrspace(5))
// CHECK: %[[lnot_ext:.*]] = zext i1 %[[lnot]] to i32
// CHECK: ret i32 %[[lnot_ext]]
int test_not_private_ptr(private char* p) {
  return !p;
}

// CHECK-LABEL: test_not_local_ptr
// SPIR64: %[[lnot:.*]] = icmp eq ptr addrspace(3) %p, addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: %[[lnot:.*]] = icmp eq ptr addrspace(3) %p, addrspacecast (ptr null to ptr addrspace(3))
// CHECK: %[[lnot_ext:.*]] = zext i1 %[[lnot]] to i32
// CHECK: ret i32 %[[lnot_ext]]
int test_not_local_ptr(local char* p) {
  return !p;
}


// CHECK-LABEL: test_and_ptr
// SPIR64: %[[tobool:.*]] = icmp ne ptr %p1, addrspacecast (ptr addrspace(4) null to ptr)
// SPIR64: %[[tobool1:.*]] = icmp ne ptr addrspace(3) %p2, addrspacecast (ptr addrspace(4) null to ptr addrspace(3))
// AMDGCN: %[[tobool:.*]] = icmp ne ptr addrspace(5) %p1, addrspacecast (ptr null to ptr addrspace(5))
// AMDGCN: %[[tobool1:.*]] = icmp ne ptr addrspace(3) %p2, addrspacecast (ptr null to ptr addrspace(3))
// CHECK: %[[res:.*]] = select i1 %[[tobool]], i1 %[[tobool1]], i1 false
// CHECK: %[[land_ext:.*]] = zext i1 %[[res]] to i32
// CHECK: ret i32 %[[land_ext]]
int test_and_ptr(private char* p1, local char* p2) {
  return p1 && p2;
}

// Test folding of null pointer in function scope.
// CHECK-NOOPT-LABEL: test_fold_private
// SPIR64-NOOPT:  call{{.*}} void @test_fold_callee
// SPIR64-NOOPT:  store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr %glob{{.*}}, align 8
// SPIR64-NOOPT:  %{{.*}} = sub i64 %{{.*}}, ptrtoint (ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)) to i64)
// AMDGCN-NOOPT: store ptr addrspace(1) null, ptr addrspace(5) %glob{{.*}}, align 8
// AMDGCN-NOOPT: %{{.*}} = sub i64 %{{.*}}, 0
// SPIR64-NOOPT:  call{{.*}} void @test_fold_callee
// SPIR64-NOOPT:  %[[SEXT:.*]] = sext i32 ptrtoint (ptr addrspacecast (ptr addrspace(4) null to ptr) to i32) to i64
// AMDGCN-NOOPT: %[[SEXT:.*]] = sext i32 ptrtoint (ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)) to i32) to i64
// CHECK-NOOPT: %{{.*}} = add nsw i64 %{{.*}}, %[[SEXT]]
// CHECK-NOOPT: %{{.*}} = sub nsw i64 %{{.*}}, 1
void test_fold_callee(void);
void test_fold_private(void) {
  global int* glob = (test_fold_callee(), (global int*)(generic char*)0);
  long x = glob - (global int*)(generic char*)0;
  x = x + (int)(test_fold_callee(), (private int*)(generic char*)(global short*)0);
  x = x - (int)((private int*)0 == (private int*)(generic char*)0);
}

// CHECK-NOOPT-LABEL: test_fold_local
// CHECK-NOOPT:  call{{.*}} void @test_fold_callee
// SPIR64-NOOPT: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr %glob{{.*}}, align 8
// SPIR64-NOOPT: %{{.*}} = sub i64 %{{.*}}, ptrtoint (ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)) to i64)
// AMDGCN-NOOPT: store ptr addrspace(1) null, ptr addrspace(5) %glob{{.*}}, align 8
// AMDGCN-NOOPT: %{{.*}} = sub i64 %{{.*}}, 0
// CHECK-NOOPT:  call{{.*}} void @test_fold_callee
// SPIR64-NOOPT: %[[SEXT:.*]] = sext i32 ptrtoint (ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)) to i32) to i64
// AMDGCN-NOOPT: %[[SEXT:.*]] = sext i32 ptrtoint (ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)) to i32) to i64
// CHECK-NOOPT: %{{.*}} = add nsw i64 %{{.*}}, %[[SEXT]]
// CHECK-NOOPT: %{{.*}} = sub nsw i64 %{{.*}}, 1
void test_fold_local(void) {
  global int* glob = (test_fold_callee(), (global int*)(generic char*)0);
  long x = glob - (global int*)(generic char*)0;
  x = x + (int)(test_fold_callee(), (local int*)(generic char*)(global short*)0);
  x = x - (int)((local int*)0 == (local int*)(generic char*)0);
}
