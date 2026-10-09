// Test the return of composite values under -freg-struct-return.

// RUN: %clang_cc1 -triple s390x-linux-gnu -freg-struct-return -emit-llvm \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -freg-struct-return -target-cpu z13 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,VECTOR
// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=MEMORY
// RUN: %clang_cc1 -triple s390x-linux-gnu -fpcc-struct-return -emit-llvm \
// RUN:   -o - %s | FileCheck %s --check-prefix=MEMORY

struct agg_1byte { char a[1]; };
struct agg_1byte ret_agg_1byte(void) { struct agg_1byte r; return r; }
// CHECK-LABEL: define{{.*}} noext i8 @ret_agg_1byte()

struct agg_3byte { char a[3]; };
struct agg_3byte ret_agg_3byte(void) { struct agg_3byte r; return r; }
// CHECK-LABEL: define{{.*}} noext i24 @ret_agg_3byte()

struct agg_8byte { char a[8]; };
struct agg_8byte ret_agg_8byte(void) { struct agg_8byte r; return r; }
// CHECK-LABEL: define{{.*}} noext i64 @ret_agg_8byte()
// MEMORY-LABEL: define{{.*}} void @ret_agg_8byte(ptr dead_on_unwind noalias writable sret(%struct.agg_8byte) align 1 %{{.*}})

struct agg_9byte { char a[9]; };
struct agg_9byte ret_agg_9byte(void) { struct agg_9byte r; return r; }
// CHECK-LABEL: define{{.*}} { i64, i8 } @ret_agg_9byte()

struct agg_16byte { char a[16]; };
struct agg_16byte ret_agg_16byte(void) { struct agg_16byte r; return r; }
// CHECK-LABEL: define{{.*}} { i64, i64 } @ret_agg_16byte()
// MEMORY-LABEL: define{{.*}} void @ret_agg_16byte(ptr dead_on_unwind noalias writable sret(%struct.agg_16byte) align 1 %{{.*}})

struct agg_17byte { char a[17]; };
struct agg_17byte ret_agg_17byte(void) { struct agg_17byte r; return r; }
// CHECK-LABEL: define{{.*}} void @ret_agg_17byte(ptr dead_on_unwind noalias writable sret(%struct.agg_17byte) align 1 %{{.*}})

struct agg_empty { };
struct agg_empty ret_agg_empty(void) { struct agg_empty r; return r; }
// CHECK-LABEL: define{{.*}} void @ret_agg_empty()
// MEMORY-LABEL: define{{.*}} void @ret_agg_empty(ptr dead_on_unwind noalias writable sret(%struct.agg_empty) align 1 %{{.*}})

struct agg_zero_array { int a[0]; };
struct agg_zero_array ret_agg_zero_array(void) { struct agg_zero_array r; return r; }
// CHECK-LABEL: define{{.*}} void @ret_agg_zero_array()

struct agg_empty get_agg_empty(void);
void call_agg_empty(void) { get_agg_empty(); }
// CHECK-LABEL: define{{.*}} void @call_agg_empty()
// CHECK: call void @get_agg_empty()

struct agg_double { double a; };
struct agg_double ret_agg_double(void) { struct agg_double r; return r; }
// CHECK-LABEL: define{{.*}} noext i64 @ret_agg_double()

_Complex int ret_complex_int(void) { return 0; }
// CHECK-LABEL: define{{.*}} void @ret_complex_int(ptr dead_on_unwind noalias writable sret({ i32, i32 }) align 4 %{{.*}})

typedef __attribute__((vector_size(16))) int v4i32;
v4i32 ret_v4i32(void) { return (v4i32){ 0 }; }
// VECTOR-LABEL: define{{.*}} <4 x i32> @ret_v4i32()
