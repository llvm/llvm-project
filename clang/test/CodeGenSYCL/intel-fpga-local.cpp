// RUN: %clang_cc1 -x c++ -triple spir64-unknown-linux-sycldevice -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

//CHECK: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}{memory:DEFAULT}{numbanks:4}
//CHECK: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}{register:1}
//CHECK: [[ANN3:@.str[\.]*[0-9]*]] = {{.*}}{memory:DEFAULT}
//CHECK: [[ANN4:@.str[\.]*[0-9]*]] = {{.*}}{memory:DEFAULT}{bankwidth:4}
//CHECK: [[ANN5:@.str[\.]*[0-9]*]] = {{.*}}{memory:BLOCK_RAM}
//CHECK: [[ANN6:@.str[\.]*[0-9]*]] = {{.*}}{memory:MLAB}
//CHECK: [[ANN7:@.str[\.]*[0-9]*]] = {{.*}}{memory:DEFAULT}{bankwidth:8}

void foo() {
  //CHECK: %[[VAR_ONE:[0-9]+]] = bitcast{{.*}}var_one
  //CHECK: %[[VAR_ONE1:var_one[0-9]+]] = bitcast{{.*}}var_one
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_ONE1]],{{.*}}[[ANN1]]
  int __attribute__((numbanks(4))) var_one;
  //CHECK: %[[VAR_TWO:[0-9]+]] = bitcast{{.*}}var_two
  //CHECK: %[[VAR_TWO1:var_two[0-9]+]] = bitcast{{.*}}var_two
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_TWO1]],{{.*}}[[ANN2]]
  int __attribute__((register)) var_two;
  //CHECK: %[[VAR_THREE:[0-9]+]] = bitcast{{.*}}var_three
  //CHECK: %[[VAR_THREE1:var_three[0-9]+]] = bitcast{{.*}}var_three
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_THREE1]],{{.*}}[[ANN3]]
  int __attribute__((__memory__)) var_three;
  //CHECK: %[[VAR_FOUR:[0-9]+]] = bitcast{{.*}}var_four
  //CHECK: %[[VAR_FOUR1:var_four[0-9]+]] = bitcast{{.*}}var_four
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_FOUR1]],{{.*}}[[ANN4]]
  int __attribute__((__bankwidth__(4))) var_four;
}

struct foo_two {
  int __attribute__((numbanks(4))) f1;
  int __attribute__((register)) f2;
  int __attribute__((__memory__)) f3;
  int __attribute__((__bankwidth__(4))) f4;
};

void bar() {
  struct foo_two s1;
  //CHECK: %[[FIELD1:.*]] = getelementptr inbounds %struct.foo_two{{.*}}
  //CHECK: %[[CAST:.*]] = bitcast{{.*}}%[[FIELD1]]
  //CHECK: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST]]{{.*}}[[ANN1]]
  s1.f1 = 0;
  //CHECK: %[[FIELD2:.*]] = getelementptr inbounds %struct.foo_two{{.*}}
  //CHECK: %[[CAST:.*]] = bitcast{{.*}}%[[FIELD2]]
  //CHECK: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST]]{{.*}}[[ANN2]]
  s1.f2 = 0;
  //CHECK: %[[FIELD3:.*]] = getelementptr inbounds %struct.foo_two{{.*}}
  //CHECK: %[[CAST:.*]] = bitcast{{.*}}%[[FIELD3]]
  //CHECK: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST]]{{.*}}[[ANN3]]
  s1.f3 = 0;
  //CHECK: %[[FIELD4:.*]] = getelementptr inbounds %struct.foo_two{{.*}}
  //CHECK: %[[CAST:.*]] = bitcast{{.*}}%[[FIELD4]]
  //CHECK: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST]]{{.*}}[[ANN4]]
  s1.f4 = 0;
}

void baz() {
  //CHECK: %[[V_ONE:[0-9]+]] = bitcast{{.*}}v_one
  //CHECK: %[[V_ONE1:v_one[0-9]+]] = bitcast{{.*}}v_one
  //CHECK: llvm.var.annotation{{.*}}%[[V_ONE1]],{{.*}}[[ANN1]]
  int v_one [[intelfpga::numbanks(4)]];
  //CHECK: %[[V_TWO:[0-9]+]] = bitcast{{.*}}v_two
  //CHECK: %[[V_TWO1:v_two[0-9]+]] = bitcast{{.*}}v_two
  //CHECK: llvm.var.annotation{{.*}}%[[V_TWO1]],{{.*}}[[ANN2]]
  int v_two [[intelfpga::register]];
  //CHECK: %[[V_THREE:[0-9]+]] = bitcast{{.*}}v_three
  //CHECK: %[[V_THREE1:v_three[0-9]+]] = bitcast{{.*}}v_three
  //CHECK: llvm.var.annotation{{.*}}%[[V_THREE1]],{{.*}}[[ANN3]]
  int v_three [[intelfpga::memory]];
  //CHECK: %[[V_FOUR:[0-9]+]] = bitcast{{.*}}v_four
  //CHECK: %[[V_FOUR1:v_four[0-9]+]] = bitcast{{.*}}v_four
  //CHECK: llvm.var.annotation{{.*}}%[[V_FOUR1]],{{.*}}[[ANN5]]
  int v_four [[intelfpga::memory("BLOCK_RAM")]];
  //CHECK: %[[V_FIVE:[0-9]+]] = bitcast{{.*}}v_five
  //CHECK: %[[V_FIVE1:v_five[0-9]+]] = bitcast{{.*}}v_five
  //CHECK: llvm.var.annotation{{.*}}%[[V_FIVE1]],{{.*}}[[ANN6]]
  int v_five [[intelfpga::memory("MLAB")]];
  //CHECK: %[[V_SIX:[0-9]+]] = bitcast{{.*}}v_six
  //CHECK: %[[V_SIX1:v_six[0-9]+]] = bitcast{{.*}}v_six
  //CHECK: llvm.var.annotation{{.*}}%[[V_SIX1]],{{.*}}[[ANN7]]
  int v_six [[intelfpga::bankwidth(8)]];
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
    bar();
    baz();
  });
  return 0;
}