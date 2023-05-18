// RUN: %clang_cc1 -triple mipsel-unknown-linux -emit-llvm -o - %s | FileCheck %s

void __attribute__ ((interrupt("vector=sw0")))
isr_sw0 (void)
{
  // CHECK: define{{.*}} void @isr_sw0() [[SW0:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=sw1")))
isr_sw1 (void)
{
  // CHECK: define{{.*}} void @isr_sw1() [[SW1:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw0")))
isr_hw0 (void)
{
  // CHECK: define{{.*}} void @isr_hw0() [[HW0:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw1")))
isr_hw1 (void)
{
  // CHECK: define{{.*}} void @isr_hw1() [[HW1:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw2")))
isr_hw2 (void)
{
  // CHECK: define{{.*}} void @isr_hw2() [[HW2:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw3")))
isr_hw3 (void)
{
  // CHECK: define{{.*}} void @isr_hw3() [[HW3:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw4")))
isr_hw4 (void)
{
  // CHECK: define{{.*}} void @isr_hw4() [[HW4:#[0-9]+]]
}

void __attribute__ ((interrupt("vector=hw5")))
isr_hw5 (void)
{
  // CHECK: define{{.*}} void @isr_hw5() [[HW5:#[0-9]+]]
}

void __attribute__ ((interrupt))
isr_eic (void)
{
  // CHECK: define{{.*}} void @isr_eic() [[EIC:#[0-9]+]]
}
// CHECK: attributes [[SW0]] = { {{.*}} "interrupt"="sw0" {{.*}} }
// CHECK: attributes [[SW1]] = { {{.*}} "interrupt"="sw1" {{.*}} }
// CHECK: attributes [[HW0]] = { {{.*}} "interrupt"="hw0" {{.*}} }
// CHECK: attributes [[HW1]] = { {{.*}} "interrupt"="hw1" {{.*}} }
// CHECK: attributes [[HW2]] = { {{.*}} "interrupt"="hw2" {{.*}} }
// CHECK: attributes [[HW3]] = { {{.*}} "interrupt"="hw3" {{.*}} }
// CHECK: attributes [[HW4]] = { {{.*}} "interrupt"="hw4" {{.*}} }
// CHECK: attributes [[HW5]] = { {{.*}} "interrupt"="hw5" {{.*}} }
// CHECK: attributes [[EIC]] = { {{.*}} "interrupt"="eic" {{.*}} }
