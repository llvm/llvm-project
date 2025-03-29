// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

unsigned Int;

void uses() {
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_type(I) device_num(Int)
#pragma acc set default_async(Int) if (Int == 5) device_type(I) device_num(Int)
// CHECK: #pragma acc set default_async(Int) device_type(I) device_num(Int)
#pragma acc set default_async(Int) device_type(I) device_num(Int)
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_num(Int)
#pragma acc set default_async(Int) if (Int == 5) device_num(Int)
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_type(I)
#pragma acc set default_async(Int) if (Int == 5) device_type(I)
// CHECK: #pragma acc set if(Int == 5) device_type(I) device_num(Int)
#pragma acc set if (Int == 5) device_type(I) device_num(Int)
// CHECK: #pragma acc set default_async(Int)
#pragma acc set default_async(Int)
// CHECK: #pragma acc set if(Int == 5)
#pragma acc set if (Int == 5)
// CHECK: #pragma acc set device_type(I)
#pragma acc set device_type(I)
// CHECK: #pragma acc set device_num(Int)
#pragma acc set device_num(Int)
}
