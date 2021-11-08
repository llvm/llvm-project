// RUN: %clang_cc1 -triple i386-apple-ios -fblocks -emit-llvm -o - %s | FileCheck %s

void __assert_rtn(const char *, const char *, int, const char *)
    __attribute__ (( noreturn ));

void (^mangle(void))(void) {
  return ^{
    void (^block)(void) = ^{
      __assert_rtn(__func__, __FILE__, __LINE__, "mangle");
    };
  };
}

// CHECK: @__func__.__mangle_block_invoke_2 = private unnamed_addr constant [22 x i8] c"mangle_block_invoke_2\00", align 1
// CHECK: @.str{{.*}} = private unnamed_addr constant {{.*}}, align 1
// CHECK: @.str[[STR1:.*]] = private unnamed_addr constant [7 x i8] c"mangle\00", align 1

// CHECK: define internal void @__mangle_block_invoke(i8* %.block_descriptor)

// CHECK: define internal void @__mangle_block_invoke_2(i8* %.block_descriptor){{.*}}{
// CHECK:   call void @__assert_rtn(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @__func__.__mangle_block_invoke_2, i32 0, i32 0), i8* getelementptr inbounds {{.*}}, i32 9, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str[[STR1]], i32 0, i32 0))
// CHECK: }

