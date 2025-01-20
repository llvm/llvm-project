// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

unsigned Int;

void uses() {
// CHECK: #pragma acc shutdown device_type(*) device_num(Int) if(Int == 5)
#pragma acc shutdown device_type(*) device_num(Int) if (Int == 5)
// CHECK: #pragma acc shutdown device_type(*) device_num(Int)
#pragma acc shutdown device_type(*) device_num(Int)
// CHECK: #pragma acc shutdown device_type(*) if(Int == 5)
#pragma acc shutdown device_type(*) if (Int == 5)
// CHECK: #pragma acc shutdown device_type(SomeName)
#pragma acc shutdown device_type(SomeName)
}
