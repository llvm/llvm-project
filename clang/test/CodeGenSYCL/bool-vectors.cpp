// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | opt -asfix -S -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}
using bool1 = bool;
using bool2 = bool __attribute__((ext_vector_type(2)));
using bool3 = bool __attribute__((ext_vector_type(3)));
using bool4 = bool __attribute__((ext_vector_type(4)));
using bool8 = bool __attribute__((ext_vector_type(8)));
using bool16 = bool __attribute__((ext_vector_type(16)));

extern bool1 foo1();
// CHECK-DAG: declare spir_func zeroext i1 @[[FOO1:[a-zA-Z0-9_]+]]()
extern bool2 foo2();
// CHECK-DAG: declare spir_func <2 x i1> @[[FOO2:[a-zA-Z0-9_]+]]()
extern bool3 foo3();
// CHECK-DAG: declare spir_func <3 x i1> @[[FOO3:[a-zA-Z0-9_]+]]()
extern bool4 foo4();
// CHECK-DAG: declare spir_func <4 x i1> @[[FOO4:[a-zA-Z0-9_]+]]()
extern bool8 foo8();
// CHECK-DAG: declare spir_func <8 x i1> @[[FOO8:[a-zA-Z0-9_]+]]()
extern bool16 foo16();
// CHECK-DAG: declare spir_func <16 x i1> @[[FOO16:[a-zA-Z0-9_]+]]()

void bar (bool1 b) {};
// CHECK-DAG: define spir_func void @[[BAR1:[a-zA-Z0-9_]+]](i1 zeroext %
void bar (bool2 b) {};
// CHECK-DAG: define spir_func void @[[BAR2:[a-zA-Z0-9_]+]](<2 x i1> %
void bar (bool3 b) {};
// CHECK-DAG: define spir_func void @[[BAR3:[a-zA-Z0-9_]+]](<3 x i1> %
void bar (bool4 b) {};
// CHECK-DAG: define spir_func void @[[BAR4:[a-zA-Z0-9_]+]](<4 x i1> %
void bar (bool8 b) {};
// CHECK-DAG: define spir_func void @[[BAR8:[a-zA-Z0-9_]+]](<8 x i1> %
void bar (bool16 b) {};
// CHECK-DAG: define spir_func void @[[BAR16:[a-zA-Z0-9_]+]](<16 x i1> %

int main() {
  kernel<class kernel_function>(
      [=]() {
        bool1 b1 = foo1();
        // CHECK-DAG: [[B1:%[a-zA-Z0-9_]+]] = alloca i8, align 1
        // CHECK-DAG: [[CALL1:%[a-zA-Z0-9_]+]] = call spir_func zeroext i1 @[[FOO1]]
        bar(b1);
        // CHECK-DAG: call spir_func void @[[BAR1]](i1 zeroext [[B1_ARG:%[a-zA-Z0-9_]+]]
        bool2 b2 = foo2();
        // CHECK-DAG: [[B2:%[a-zA-Z0-9_]+]] = alloca <2 x i1>, align 2
        // CHECK-DAG: [[CALL2:%[a-zA-Z0-9_]+]] = call spir_func <2 x i1> @[[FOO2]]
        bar(b2);
        // CHECK-DAG: call spir_func void @[[BAR2]](<2 x i1> [[B2_ARG:%[a-zA-Z0-9_]+]]
        bool3 b3 = foo3();
        // CHECK-DAG: [[B3:%[a-zA-Z0-9_]+]] = alloca <3 x i1>, align 4
        // CHECK-DAG: [[CALL3:%[a-zA-Z0-9_]+]] = call spir_func <3 x i1> @[[FOO3]]
        bar(b3);
        // CHECK-DAG: call spir_func void @[[BAR3]](<3 x i1> [[B3_ARG:%[a-zA-Z0-9_]+]]
        bool4 b4 = foo4();
        // CHECK-DAG: [[B4:%[a-zA-Z0-9_]+]] = alloca <4 x i1>, align 4
        // CHECK-DAG: [[CALL4:%[a-zA-Z0-9_]+]] = call spir_func <4 x i1> @[[FOO4]]
        bar(b4);
        // CHECK-DAG: call spir_func void @[[BAR4]](<4 x i1> [[B4_ARG:%[a-zA-Z0-9_]+]]
        bool8 b8 = foo8();
        // CHECK-DAG: [[B8:%[a-zA-Z0-9_]+]] = alloca <8 x i1>, align 8
        // CHECK-DAG: [[CALL8:%[a-zA-Z0-9_]+]] = call spir_func <8 x i1> @[[FOO8]]
        bar(b8);
        // CHECK-DAG: call spir_func void @[[BAR8]](<8 x i1> [[B8_ARG:%[a-zA-Z0-9_]+]]
        bool16 b16 = foo16();
        // CHECK-DAG: [[B16:%[a-zA-Z0-9_]+]] = alloca <16 x i1>, align 16
        // CHECK-DAG: [[CALL16:%[a-zA-Z0-9_]+]] = call spir_func <16 x i1> @[[FOO16]]
        bar(b16);
        // CHECK-DAG: call spir_func void @[[BAR16]](<16 x i1> [[B16_ARG:%[a-zA-Z0-9_]+]]
      });
  return 0;
}
