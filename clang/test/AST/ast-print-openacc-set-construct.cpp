// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

unsigned Int;

void uses() {
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_type(default) device_num(Int)
#pragma acc set default_async(Int) if (Int == 5) device_type(default) device_num(Int)
// CHECK: #pragma acc set default_async(Int) device_type(nvidia) device_num(Int)
#pragma acc set default_async(Int) device_type(nvidia) device_num(Int)
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_num(Int)
#pragma acc set default_async(Int) if (Int == 5) device_num(Int)
// CHECK: #pragma acc set default_async(Int) if(Int == 5) device_type(host)
#pragma acc set default_async(Int) if (Int == 5) device_type(host)
// CHECK: #pragma acc set if(Int == 5) device_type(multicore) device_num(Int)
#pragma acc set if (Int == 5) device_type(multicore) device_num(Int)
// CHECK: #pragma acc set default_async(Int)
#pragma acc set default_async(Int)
// CHECK: #pragma acc set if(Int == 5) device_type(multicore)
#pragma acc set if (Int == 5) device_type(multicore)
// CHECK: #pragma acc set device_type(radeon)
#pragma acc set device_type(radeon)
// CHECK: #pragma acc set device_num(Int)
#pragma acc set device_num(Int)
}
