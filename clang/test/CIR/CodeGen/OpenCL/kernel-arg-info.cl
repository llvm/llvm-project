// See also clang/test/CodeGenOpenCL/kernel-arg-info.cl
// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-cir -o - -triple spirv64-unknown-unknown -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-cir -o - -triple spirv64-unknown-unknown -cl-kernel-arg-info -o %t.arginfo.cir
// RUN: FileCheck %s --input-file=%t.arginfo.cir --check-prefix=CIR-ARGINFO

// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-llvm -o - -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-llvm -o - -triple spirv64-unknown-unknown -cl-kernel-arg-info -o %t.arginfo.ll
// RUN: FileCheck %s --input-file=%t.arginfo.ll --check-prefix=LLVM-ARGINFO

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
// CIR-DAG: #fn_attr[[KERNEL0:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], access_qual = ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none"], type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int", "int", "int", "int"], base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int", "int", "int", "int"], type_qual = ["", "restrict", "const", "restrict const", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile", "", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile", "", "", "", ""]>
// CIR-DAG: cir.func @foo({{.+}}) extra(#fn_attr[[KERNEL0]])
// CIR-ARGINFO-DAG: #fn_attr[[KERNEL0:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], access_qual = ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none"], type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int", "int", "int", "int"], base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*", "int", "int", "int", "int"], type_qual = ["", "restrict", "const", "restrict const", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile", "", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile", "", "", "", ""], name = ["globalintp", "globalintrestrictp", "globalconstintp", "globalconstintrestrictp", "constantintp", "constantintrestrictp", "globalconstvolatileintp", "globalconstvolatileintrestrictp", "globalvolatileintp", "globalvolatileintrestrictp", "localintp", "localintrestrictp", "localconstintp", "localconstintrestrictp", "localconstvolatileintp", "localconstvolatileintrestrictp", "localvolatileintp", "localvolatileintrestrictp", "X", "constint", "constvolatileint", "volatileint"]>
// CIR-ARGINFO-DAG: cir.func @foo({{.+}}) extra(#fn_attr[[KERNEL0]])

// LLVM-DAG: define{{.*}} void @foo{{.+}} !kernel_arg_addr_space ![[MD11:[0-9]+]] !kernel_arg_access_qual ![[MD12:[0-9]+]] !kernel_arg_type ![[MD13:[0-9]+]] !kernel_arg_base_type ![[MD13]] !kernel_arg_type_qual ![[MD14:[0-9]+]] {
// LLVM-ARGINFO-DAG: define{{.*}} void @foo{{.+}} !kernel_arg_addr_space ![[MD11:[0-9]+]] !kernel_arg_access_qual ![[MD12:[0-9]+]] !kernel_arg_type ![[MD13:[0-9]+]] !kernel_arg_base_type ![[MD13]] !kernel_arg_type_qual ![[MD14:[0-9]+]] !kernel_arg_name ![[MD15:[0-9]+]] {

// LLVM-DAG: ![[MD11]] = !{i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 1, i32 1, i32 1, i32 1, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0}
// LLVM-DAG: ![[MD12]] = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
// LLVM-DAG: ![[MD13]] = !{!"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int", !"int", !"int", !"int"}
// LLVM-DAG: ![[MD14]] = !{!"", !"restrict", !"const", !"restrict const", !"const", !"restrict const", !"const volatile", !"restrict const volatile", !"volatile", !"restrict volatile", !"", !"restrict", !"const", !"restrict const", !"const volatile", !"restrict const volatile", !"volatile", !"restrict volatile", !"", !"", !"", !""}
// LLVM-ARGINFO-DAG: ![[MD15]] = !{!"globalintp", !"globalintrestrictp", !"globalconstintp", !"globalconstintrestrictp", !"constantintp", !"constantintrestrictp", !"globalconstvolatileintp", !"globalconstvolatileintrestrictp", !"globalvolatileintp", !"globalvolatileintrestrictp", !"localintp", !"localintrestrictp", !"localconstintp", !"localconstintrestrictp", !"localconstvolatileintp", !"localconstvolatileintrestrictp", !"localvolatileintp", !"localvolatileintrestrictp", !"X", !"constint", !"constvolatileint", !"volatileint"}

typedef unsigned int myunsignedint;
kernel void foo4(__global unsigned int * X, __global myunsignedint * Y) {
}

// CIR-DAG: #fn_attr[[KERNEL4:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32, 1 : i32], access_qual = ["none", "none"], type = ["uint*", "myunsignedint*"], base_type = ["uint*", "uint*"], type_qual = ["", ""]>
// CIR-DAG: cir.func @foo4({{.+}}) extra(#fn_attr[[KERNEL4]])
// CIR-ARGINFO-DAG: #fn_attr[[KERNEL4:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32, 1 : i32], access_qual = ["none", "none"], type = ["uint*", "myunsignedint*"], base_type = ["uint*", "uint*"], type_qual = ["", ""], name = ["X", "Y"]>
// CIR-ARGINFO-DAG: cir.func @foo4({{.+}}) extra(#fn_attr[[KERNEL4]])

// LLVM-DAG: define{{.*}} void @foo4{{.+}} !kernel_arg_addr_space ![[MD41:[0-9]+]] !kernel_arg_access_qual ![[MD42:[0-9]+]] !kernel_arg_type ![[MD43:[0-9]+]] !kernel_arg_base_type ![[MD44:[0-9]+]] !kernel_arg_type_qual ![[MD45:[0-9]+]] {
// LLVM-ARGINFO-DAG: define{{.*}} void @foo4{{.+}} !kernel_arg_addr_space ![[MD41:[0-9]+]] !kernel_arg_access_qual ![[MD42:[0-9]+]] !kernel_arg_type ![[MD43:[0-9]+]] !kernel_arg_base_type ![[MD44:[0-9]+]] !kernel_arg_type_qual ![[MD45:[0-9]+]] !kernel_arg_name ![[MD46:[0-9]+]] {

// LLVM-DAG: ![[MD41]] = !{i32 1, i32 1}
// LLVM-DAG: ![[MD42]] = !{!"none", !"none"}
// LLVM-DAG: ![[MD43]] = !{!"uint*", !"myunsignedint*"}
// LLVM-DAG: ![[MD44]] = !{!"uint*", !"uint*"}
// LLVM-DAG: ![[MD45]] = !{!"", !""}
// LLVM-ARGINFO-DAG: ![[MD46]] = !{!"X", !"Y"}

typedef char char16 __attribute__((ext_vector_type(16)));
__kernel void foo6(__global char16 arg[]) {}

// CIR-DAG: #fn_attr[[KERNEL6:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32], access_qual = ["none"], type = ["char16*"], base_type = ["char __attribute__((ext_vector_type(16)))*"], type_qual = [""]>
// CIR-DAG: cir.func @foo6({{.+}}) extra(#fn_attr[[KERNEL6]])
// CIR-ARGINFO-DAG: #fn_attr[[KERNEL6:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32], access_qual = ["none"], type = ["char16*"], base_type = ["char __attribute__((ext_vector_type(16)))*"], type_qual = [""], name = ["arg"]>
// CIR-ARGINFO-DAG: cir.func @foo6({{.+}}) extra(#fn_attr[[KERNEL6]])

// LLVM-DAG: !kernel_arg_type ![[MD61:[0-9]+]]
// LLVM-ARGINFO-DAG: !kernel_arg_name ![[MD62:[0-9]+]]
// LLVM-DAG: ![[MD61]] = !{!"char16*"}
// LLVM-ARGINFO-DAG: ![[MD62]] = !{!"arg"}

kernel void foo9(signed char sc1,  global const signed char* sc2) {}

// CIR-DAG: #fn_attr[[KERNEL9:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [0 : i32, 1 : i32], access_qual = ["none", "none"], type = ["char", "char*"], base_type = ["char", "char*"], type_qual = ["", "const"]>
// CIR-DAG: cir.func @foo9({{.+}}) extra(#fn_attr[[KERNEL9]])
// CIR-ARGINFO-DAG: #fn_attr[[KERNEL9:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [0 : i32, 1 : i32], access_qual = ["none", "none"], type = ["char", "char*"], base_type = ["char", "char*"], type_qual = ["", "const"], name = ["sc1", "sc2"]>
// CIR-ARGINFO-DAG: cir.func @foo9({{.+}}) extra(#fn_attr[[KERNEL9]])

// LLVM-DAG: define{{.*}} void @foo9{{.+}} !kernel_arg_addr_space ![[SCHAR_AS_QUAL:[0-9]+]] !kernel_arg_access_qual ![[MD42]] !kernel_arg_type ![[SCHAR_TY:[0-9]+]] !kernel_arg_base_type ![[SCHAR_TY]] !kernel_arg_type_qual ![[SCHAR_QUAL:[0-9]+]] {
// LLVM-ARGINFO-DAG: define{{.*}} void @foo9{{.+}} !kernel_arg_addr_space ![[SCHAR_AS_QUAL:[0-9]+]] !kernel_arg_access_qual ![[MD42]] !kernel_arg_type ![[SCHAR_TY:[0-9]+]] !kernel_arg_base_type ![[SCHAR_TY]] !kernel_arg_type_qual ![[SCHAR_QUAL:[0-9]+]] !kernel_arg_name ![[SCHAR_ARG_NAMES:[0-9]+]] {

// LLVM-DAG: ![[SCHAR_AS_QUAL]] = !{i32 0, i32 1}
// LLVM-DAG: ![[SCHAR_TY]] = !{!"char", !"char*"}
// LLVM-DAG: ![[SCHAR_QUAL]] = !{!"", !"const"}
// LLVM-ARGINFO-DAG: ![[SCHAR_ARG_NAMES]] = !{!"sc1", !"sc2"}
