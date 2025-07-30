// RUN: %clang_cc1 %s -emit-llvm -fextend-variable-liveness -o - | FileCheck %s --implicit-check-not=fake.use
// Make sure we don't generate fake.use for non-scalar variables, unless they
// are small enough that they may be represented as a scalar in LLVM IR.
// Make sure we don't generate fake.use for volatile variables
// and parameters even when they are scalar.

struct BigAggr {
  unsigned long t;
  char c[1024];
  unsigned char r[32];
};

struct SmallAggr {
  int i;
  int j;
};

int foo(volatile int vol_param, int param)
{
  struct BigAggr big;
  struct SmallAggr small;
  volatile int vol_local;
  int local;
  unsigned long_arr[5];
  unsigned short_arr[4];
  return 0;
}

// CHECK: [[SMALL_ARR_FAKE_USE:%.+]] = load [4 x i[[#UINT_SIZE:]]], ptr %short_arr
// CHECK: call void (...) @llvm.fake.use([4 x i[[#UINT_SIZE]]] [[SMALL_ARR_FAKE_USE]])

// CHECK: [[LOCAL_FAKE_USE:%.+]] = load i32, ptr %local
// CHECK: call void (...) @llvm.fake.use(i32 [[LOCAL_FAKE_USE]])

// CHECK: [[SMALL_FAKE_USE:%.+]] = load %struct.SmallAggr, ptr %small
// CHECK: call void (...) @llvm.fake.use(%struct.SmallAggr [[SMALL_FAKE_USE]])

// CHECK: [[PARAM_FAKE_USE:%.+]] = load i32, ptr %param.addr
// CHECK: call void (...) @llvm.fake.use(i32 [[PARAM_FAKE_USE]])

// CHECK: declare void @llvm.fake.use
