// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown -cl-kernel-arg-info | FileCheck %s -check-prefix ARGINFO

kernel void foo(__global int * restrict X, const int Y, 
                volatile int anotherArg, __constant float * restrict Z) {
  *X = Y + anotherArg;
}
// CHECK: define spir_kernel void @foo{{[^!]+}}
// CHECK: !kernel_arg_addr_space ![[MD11:[0-9]+]]
// CHECK: !kernel_arg_access_qual ![[MD12:[0-9]+]]
// CHECK: !kernel_arg_type ![[MD13:[0-9]+]]
// CHECK: !kernel_arg_base_type ![[MD13]]
// CHECK: !kernel_arg_type_qual ![[MD14:[0-9]+]]
// CHECK-NOT: !kernel_arg_name
// ARGINFO: !kernel_arg_name ![[MD15:[0-9]+]]

kernel void foo2(read_only image1d_t img1, image2d_t img2, write_only image2d_array_t img3) {
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

// CHECK: ![[MD11]] = !{i32 1, i32 0, i32 0, i32 2}
// CHECK: ![[MD12]] = !{!"none", !"none", !"none", !"none"}
// CHECK: ![[MD13]] = !{!"int*", !"int", !"int", !"float*"}
// CHECK: ![[MD14]] = !{!"restrict", !"const", !"volatile", !"restrict const"}
// ARGINFO: ![[MD15]] = !{!"X", !"Y", !"anotherArg", !"Z"}
// CHECK: ![[MD21]] = !{i32 1, i32 1, i32 1}
// CHECK: ![[MD22]] = !{!"read_only", !"read_only", !"write_only"}
// CHECK: ![[MD23]] = !{!"__read_only image1d_t", !"__read_only image2d_t", !"__write_only image2d_array_t"}
// CHECK: ![[MD24]] = !{!"", !"", !""}
// ARGINFO: ![[MD25]] = !{!"img1", !"img2", !"img3"}
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
// CHECK: ![[MD52]] = !{!"myImage", !"__write_only image1d_t"}
// CHECK: ![[MD53]] = !{!"__read_only image1d_t", !"__write_only image1d_t"}
// ARGINFO: ![[MD54]] = !{!"img1", !"img2"}

