// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -triple spir-unknown-unknown -cl-kernel-arg-info | FileCheck %s -check-prefix ARGINFO

kernel void foo(global int * globalintp, global int * restrict globalintrestrictp,
                global const int * globalconstintp,
                global const int * restrict globalconstintrestrictp,
                constant int * constantintp, constant int * restrict constantintrestrictp,
                global const volatile int * globalconstvolatileintp,
                global const volatile int * restrict globalconstvolatileintrestrictp,
                global volatile int * globalvolatileintp,
                global volatile int * restrict globalvolatileintrestrictp,
                local int * localintp, local int * restrict localintrestrictp,
                local const int * localconstintp,
                local const int * restrict localconstintrestrictp,
                local const volatile int * localconstvolatileintp,
                local const volatile int * restrict localconstvolatileintrestrictp,
                local volatile int * localvolatileintp,
                local volatile int * restrict localvolatileintrestrictp,
                int X, const int constint, const volatile int constvolatileint,
                volatile int volatileint) {
  *globalintrestrictp = constint + volatileint;
}
// CHECK: define spir_kernel void @foo{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD11:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD12:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD13:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD13]]
// CHECK: !kernel_arg_type_qual ![[MD14:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD15:[0-9]+]]

kernel void foo2(read_only image1d_t img1, image2d_t img2, write_only image2d_array_t img3, read_write image1d_t img4) {
}
// CHECK: define spir_kernel void @foo2{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD21:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD22:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD23:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD23]]
// CHECK: !kernel_arg_type_qual ![[MD24:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD25:[0-9]+]]

kernel void foo3(__global half * X) {
}
// CHECK: define spir_kernel void @foo3{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD31:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD32:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD33:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD33]]
// CHECK: !kernel_arg_type_qual ![[MD34:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD35:[0-9]+]]

typedef unsigned int myunsignedint;
kernel void foo4(__global unsigned int * X, __global myunsignedint * Y) {
}
// CHECK: define spir_kernel void @foo4{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD41:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD42:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD43:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD44:[0-9]+]]
// CHECK: !kernel_arg_type_qual ![[MD45:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD46:[0-9]+]]

typedef image1d_t myImage;
kernel void foo5(myImage img1, write_only image1d_t img2) {
}
// CHECK: define spir_kernel void @foo5{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD41:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD51:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD52:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD53:[0-9]+]]
// CHECK: !kernel_arg_type_qual ![[MD45]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD54:[0-9]+]]

typedef char char16 __attribute__((ext_vector_type(16)));
__kernel void foo6(__global char16 arg[]) {}
// CHECK: !kernel_arg_type ![[MD61:[0-9]+]]
// ARGINFO: !kernel_arg_name ![[MD62:[0-9]+]]

typedef read_only  image1d_t ROImage;
typedef write_only image1d_t WOImage;
typedef read_write image1d_t RWImage;
kernel void foo7(ROImage ro, WOImage wo, RWImage rw) {
}
// CHECK: define spir_kernel void @foo7{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD71:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD72:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD73:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD74:[0-9]+]]
// CHECK: !kernel_arg_type_qual ![[MD75:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD76:[0-9]+]]

// CHECK: ![[MD11]] = !{i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 1, i32 1, i32 1, i32 1, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0}
// CHECK: ![[MD12]] = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
// CHECK: ![[MD13]] = !{!"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int", !"int", !"int", !"int"}
// CHECK: ![[MD14]] = !{!"", !"restrict", !"const", !"restrict const", !"const", !"restrict const", !"const volatile", !"restrict const volatile", !"volatile", !"restrict volatile", !"", !"restrict", !"const", !"restrict const", !"const volatile", !"restrict const volatile", !"volatile", !"restrict volatile", !"", !"", !"", !""}
// ARGINFO: ![[MD15]] = !{!"globalintp", !"globalintrestrictp", !"globalconstintp", !"globalconstintrestrictp", !"constantintp", !"constantintrestrictp", !"globalconstvolatileintp", !"globalconstvolatileintrestrictp", !"globalvolatileintp", !"globalvolatileintrestrictp", !"localintp", !"localintrestrictp", !"localconstintp", !"localconstintrestrictp", !"localconstvolatileintp", !"localconstvolatileintrestrictp", !"localvolatileintp", !"localvolatileintrestrictp", !"X", !"constint", !"constvolatileint", !"volatileint"}
// CHECK: ![[MD21]] = !{i32 1, i32 1, i32 1, i32 1}
// CHECK: ![[MD22]] = !{!"read_only", !"read_only", !"write_only", !"read_write"}
// CHECK: ![[MD23]] = !{!"image1d_t", !"image2d_t", !"image2d_array_t", !"image1d_t"}
// CHECK: ![[MD24]] = !{!"", !"", !"", !""}
// ARGINFO: ![[MD25]] = !{!"img1", !"img2", !"img3", !"img4"}
// CHECK: ![[MD31]] = !{i32 1}
// CHECK: ![[MD32]] = !{!"none"}
// CHECK: ![[MD33]] = !{!"half*"}
// CHECK: ![[MD34]] = !{!""}
// ARGINFO: ![[MD35]] = !{!"X"}
// CHECK: ![[MD41]] = !{i32 1, i32 1}
// CHECK: ![[MD42]] = !{!"none", !"none"}
// CHECK: ![[MD43]] = !{!"uint*", !"myunsignedint*"}
// CHECK: ![[MD44]] = !{!"uint*", !"uint*"}
// CHECK: ![[MD45]] = !{!"", !""}
// ARGINFO: ![[MD46]] = !{!"X", !"Y"}
// CHECK: ![[MD51]] = !{!"read_only", !"write_only"}
// CHECK: ![[MD52]] = !{!"myImage", !"image1d_t"}
// CHECK: ![[MD53]] = !{!"image1d_t", !"image1d_t"}
// ARGINFO: ![[MD54]] = !{!"img1", !"img2"}
// CHECK: ![[MD61]] = !{!"char16*"}
// ARGINFO: ![[MD62]] = !{!"arg"}
// CHECK: ![[MD71]] = !{i32 1, i32 1, i32 1}
// CHECK: ![[MD72]] = !{!"read_only", !"write_only", !"read_write"}
// CHECK: ![[MD73]] = !{!"ROImage", !"WOImage", !"RWImage"}
// CHECK: ![[MD74]] = !{!"image1d_t", !"image1d_t", !"image1d_t"}
// CHECK: ![[MD75]] = !{!"", !"", !""}
// ARGINFO: ![[MD76]] = !{!"ro", !"wo", !"rw"}

