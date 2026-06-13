// See also clang/test/CodeGenOpenCL/kernel-arg-info.cl.
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-cir -cl-kernel-arg-info -o %t.arginfo.cir
// RUN: FileCheck %s --input-file=%t.arginfo.cir --check-prefix=CIR-ARGINFO

// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-llvm -cl-kernel-arg-info -o %t.arginfo.ll
// RUN: FileCheck %s --input-file=%t.arginfo.ll --check-prefix=LLVM-ARGINFO
// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-llvm -o %t.ogcg.ll
// RUN: FileCheck %s --input-file=%t.ogcg.ll --check-prefix=LLVM
// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-llvm -cl-kernel-arg-info -o %t.ogcg.arginfo.ll
// RUN: FileCheck %s --input-file=%t.ogcg.arginfo.ll --check-prefix=LLVM-ARGINFO

kernel void global_qualifier_kernel_args(
    global int *globalintp, global int *restrict globalintrestrictp,
    global const int *globalconstintp,
    global const int *restrict globalconstintrestrictp,
    global const volatile int *globalconstvolatileintp,
    global const volatile int *restrict globalconstvolatileintrestrictp,
    global volatile int *globalvolatileintp,
    global volatile int *restrict globalvolatileintrestrictp) {}

// CIR-LABEL: cir.func{{.*}} @global_qualifier_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>]
// CIR-SAME: access_qual = ["none", "none", "none", "none", "none", "none", "none", "none"]
// CIR-SAME: type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-SAME: base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-SAME: type_qual = ["", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile"]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @global_qualifier_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none", "none", "none", "none", "none", "none", "none"]
// CIR-ARGINFO-SAME: type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-ARGINFO-SAME: base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-ARGINFO-SAME: type_qual = ["", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile"]
// CIR-ARGINFO-SAME: name = ["globalintp", "globalintrestrictp", "globalconstintp", "globalconstintrestrictp", "globalconstvolatileintp", "globalconstvolatileintrestrictp", "globalvolatileintp", "globalvolatileintrestrictp"]

// LLVM-DAG: define{{.*}} void @global_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[GLOBAL_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[GLOBAL_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[GLOBAL_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_type_qual ![[GLOBAL_TYPE_QUALS:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @global_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[GLOBAL_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[GLOBAL_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[GLOBAL_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_type_qual ![[GLOBAL_TYPE_QUALS:[0-9]+]] !kernel_arg_name ![[GLOBAL_ARG_NAMES:[0-9]+]]
// LLVM-DAG: ![[GLOBAL_ADDR_SPACES]] = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
// LLVM-DAG: ![[GLOBAL_ACCESS_QUALS]] = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
// LLVM-DAG: ![[GLOBAL_ARG_TYPES]] = !{!"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*"}
// LLVM-DAG: ![[GLOBAL_TYPE_QUALS]] = !{!"", !"restrict", !"const", !"restrict const", !"const volatile", !"restrict const volatile", !"volatile", !"restrict volatile"}
// LLVM-ARGINFO-DAG: ![[GLOBAL_ARG_NAMES]] = !{!"globalintp", !"globalintrestrictp", !"globalconstintp", !"globalconstintrestrictp", !"globalconstvolatileintp", !"globalconstvolatileintrestrictp", !"globalvolatileintp", !"globalvolatileintrestrictp"}

kernel void constant_kernel_args(constant int *constantintp,
                                 constant int *restrict constantintrestrictp) {}

// CIR-LABEL: cir.func{{.*}} @constant_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(offload_constant)>, #cir<lang_address_space(offload_constant)>]
// CIR-SAME: access_qual = ["none", "none"]
// CIR-SAME: type = ["int*", "int*"]
// CIR-SAME: base_type = ["int*", "int*"]
// CIR-SAME: type_qual = ["const", "restrict const"]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @constant_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(offload_constant)>, #cir<lang_address_space(offload_constant)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none"]
// CIR-ARGINFO-SAME: type = ["int*", "int*"]
// CIR-ARGINFO-SAME: base_type = ["int*", "int*"]
// CIR-ARGINFO-SAME: type_qual = ["const", "restrict const"]
// CIR-ARGINFO-SAME: name = ["constantintp", "constantintrestrictp"]

// LLVM-DAG: define{{.*}} void @constant_kernel_args{{.+}} !kernel_arg_addr_space ![[CONSTANT_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[CONSTANT_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[CONSTANT_ARG_TYPES]] !kernel_arg_type_qual ![[CONSTANT_TYPE_QUALS:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @constant_kernel_args{{.+}} !kernel_arg_addr_space ![[CONSTANT_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[CONSTANT_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[CONSTANT_ARG_TYPES]] !kernel_arg_type_qual ![[CONSTANT_TYPE_QUALS:[0-9]+]] !kernel_arg_name ![[CONSTANT_ARG_NAMES:[0-9]+]]
// LLVM-DAG: ![[CONSTANT_ADDR_SPACES]] = !{i32 2, i32 2}
// LLVM-DAG: ![[CONSTANT_ACCESS_QUALS]] = !{!"none", !"none"}
// LLVM-DAG: ![[CONSTANT_ARG_TYPES]] = !{!"int*", !"int*"}
// LLVM-DAG: ![[CONSTANT_TYPE_QUALS]] = !{!"const", !"restrict const"}
// LLVM-ARGINFO-DAG: ![[CONSTANT_ARG_NAMES]] = !{!"constantintp", !"constantintrestrictp"}

kernel void local_qualifier_kernel_args(
    local int *localintp, local int *restrict localintrestrictp,
    local const int *localconstintp,
    local const int *restrict localconstintrestrictp,
    local const volatile int *localconstvolatileintp,
    local const volatile int *restrict localconstvolatileintrestrictp,
    local volatile int *localvolatileintp,
    local volatile int *restrict localvolatileintrestrictp) {}

// CIR-LABEL: cir.func{{.*}} @local_qualifier_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>]
// CIR-SAME: access_qual = ["none", "none", "none", "none", "none", "none", "none", "none"]
// CIR-SAME: type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-SAME: base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-SAME: type_qual = ["", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile"]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @local_qualifier_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>, #cir<lang_address_space(offload_local)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none", "none", "none", "none", "none", "none", "none"]
// CIR-ARGINFO-SAME: type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-ARGINFO-SAME: base_type = ["int*", "int*", "int*", "int*", "int*", "int*", "int*", "int*"]
// CIR-ARGINFO-SAME: type_qual = ["", "restrict", "const", "restrict const", "const volatile", "restrict const volatile", "volatile", "restrict volatile"]
// CIR-ARGINFO-SAME: name = ["localintp", "localintrestrictp", "localconstintp", "localconstintrestrictp", "localconstvolatileintp", "localconstvolatileintrestrictp", "localvolatileintp", "localvolatileintrestrictp"]

// LLVM-DAG: define{{.*}} void @local_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[LOCAL_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[GLOBAL_ACCESS_QUALS]] !kernel_arg_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_base_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_type_qual ![[GLOBAL_TYPE_QUALS]]
// LLVM-ARGINFO-DAG: define{{.*}} void @local_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[LOCAL_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[GLOBAL_ACCESS_QUALS]] !kernel_arg_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_base_type ![[GLOBAL_ARG_TYPES]] !kernel_arg_type_qual ![[GLOBAL_TYPE_QUALS]] !kernel_arg_name ![[LOCAL_ARG_NAMES:[0-9]+]]
// LLVM-DAG: ![[LOCAL_ADDR_SPACES]] = !{i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3}
// LLVM-ARGINFO-DAG: ![[LOCAL_ARG_NAMES]] = !{!"localintp", !"localintrestrictp", !"localconstintp", !"localconstintrestrictp", !"localconstvolatileintp", !"localconstvolatileintrestrictp", !"localvolatileintp", !"localvolatileintrestrictp"}

kernel void private_qualifier_kernel_args(int X, const int constint,
                                          const volatile int constvolatileint,
                                          volatile int volatileint) {}

// CIR-LABEL: cir.func{{.*}} @private_qualifier_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(default)>, #cir<lang_address_space(default)>, #cir<lang_address_space(default)>, #cir<lang_address_space(default)>]
// CIR-SAME: access_qual = ["none", "none", "none", "none"]
// CIR-SAME: type = ["int", "int", "int", "int"]
// CIR-SAME: base_type = ["int", "int", "int", "int"]
// CIR-SAME: type_qual = ["", "", "", ""]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @private_qualifier_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(default)>, #cir<lang_address_space(default)>, #cir<lang_address_space(default)>, #cir<lang_address_space(default)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none", "none", "none"]
// CIR-ARGINFO-SAME: type = ["int", "int", "int", "int"]
// CIR-ARGINFO-SAME: base_type = ["int", "int", "int", "int"]
// CIR-ARGINFO-SAME: type_qual = ["", "", "", ""]
// CIR-ARGINFO-SAME: name = ["X", "constint", "constvolatileint", "volatileint"]

// LLVM-DAG: define{{.*}} void @private_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[PRIVATE_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[PRIVATE_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[PRIVATE_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[PRIVATE_ARG_TYPES]] !kernel_arg_type_qual ![[PRIVATE_TYPE_QUALS:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @private_qualifier_kernel_args{{.+}} !kernel_arg_addr_space ![[PRIVATE_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[PRIVATE_ACCESS_QUALS:[0-9]+]] !kernel_arg_type ![[PRIVATE_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[PRIVATE_ARG_TYPES]] !kernel_arg_type_qual ![[PRIVATE_TYPE_QUALS:[0-9]+]] !kernel_arg_name ![[PRIVATE_ARG_NAMES:[0-9]+]]
// LLVM-DAG: ![[PRIVATE_ADDR_SPACES]] = !{i32 0, i32 0, i32 0, i32 0}
// LLVM-DAG: ![[PRIVATE_ACCESS_QUALS]] = !{!"none", !"none", !"none", !"none"}
// LLVM-DAG: ![[PRIVATE_ARG_TYPES]] = !{!"int", !"int", !"int", !"int"}
// LLVM-DAG: ![[PRIVATE_TYPE_QUALS]] = !{!"", !"", !"", !""}
// LLVM-ARGINFO-DAG: ![[PRIVATE_ARG_NAMES]] = !{!"X", !"constint", !"constvolatileint", !"volatileint"}

typedef unsigned int myunsignedint;
kernel void typedef_kernel_args(__global unsigned int *X,
                                __global myunsignedint *Y) {}

// CIR-LABEL: cir.func{{.*}} @typedef_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>]
// CIR-SAME: access_qual = ["none", "none"]
// CIR-SAME: type = ["uint*", "myunsignedint*"]
// CIR-SAME: base_type = ["uint*", "uint*"]
// CIR-SAME: type_qual = ["", ""]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @typedef_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_global)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none"]
// CIR-ARGINFO-SAME: type = ["uint*", "myunsignedint*"]
// CIR-ARGINFO-SAME: base_type = ["uint*", "uint*"]
// CIR-ARGINFO-SAME: type_qual = ["", ""]
// CIR-ARGINFO-SAME: name = ["X", "Y"]

// LLVM-DAG: define{{.*}} void @typedef_kernel_args{{.+}} !kernel_arg_addr_space ![[TYPEDEF_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS]] !kernel_arg_type ![[TYPEDEF_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[TYPEDEF_BASE_TYPES:[0-9]+]] !kernel_arg_type_qual ![[TYPEDEF_TYPE_QUALS:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @typedef_kernel_args{{.+}} !kernel_arg_addr_space ![[TYPEDEF_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS]] !kernel_arg_type ![[TYPEDEF_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[TYPEDEF_BASE_TYPES:[0-9]+]] !kernel_arg_type_qual ![[TYPEDEF_TYPE_QUALS:[0-9]+]] !kernel_arg_name ![[TYPEDEF_ARG_NAMES:[0-9]+]]

// LLVM-DAG: ![[TYPEDEF_ADDR_SPACES]] = !{i32 1, i32 1}
// LLVM-DAG: ![[TYPEDEF_ARG_TYPES]] = !{!"uint*", !"myunsignedint*"}
// LLVM-DAG: ![[TYPEDEF_BASE_TYPES]] = !{!"uint*", !"uint*"}
// LLVM-DAG: ![[TYPEDEF_TYPE_QUALS]] = !{!"", !""}
// LLVM-ARGINFO-DAG: ![[TYPEDEF_ARG_NAMES]] = !{!"X", !"Y"}

typedef char char16 __attribute__((ext_vector_type(16)));
__kernel void vector_typedef_kernel_arg(__global char16 arg[]) {}

// CIR-LABEL: cir.func{{.*}} @vector_typedef_kernel_arg
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(offload_global)>]
// CIR-SAME: access_qual = ["none"]
// CIR-SAME: type = ["char16*"]
// CIR-SAME: base_type = ["char __attribute__((ext_vector_type(16)))*"]
// CIR-SAME: type_qual = [""]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @vector_typedef_kernel_arg
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(offload_global)>]
// CIR-ARGINFO-SAME: access_qual = ["none"]
// CIR-ARGINFO-SAME: type = ["char16*"]
// CIR-ARGINFO-SAME: base_type = ["char __attribute__((ext_vector_type(16)))*"]
// CIR-ARGINFO-SAME: type_qual = [""]
// CIR-ARGINFO-SAME: name = ["arg"]

// LLVM-DAG: define{{.*}} void @vector_typedef_kernel_arg{{.+}} !kernel_arg_type ![[VECTOR_TYPEDEF_ARG_TYPES:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @vector_typedef_kernel_arg{{.+}} !kernel_arg_name ![[VECTOR_TYPEDEF_ARG_NAMES:[0-9]+]]
// LLVM-DAG: ![[VECTOR_TYPEDEF_ARG_TYPES]] = !{!"char16*"}
// LLVM-ARGINFO-DAG: ![[VECTOR_TYPEDEF_ARG_NAMES]] = !{!"arg"}

kernel void signed_char_kernel_args(signed char sc1,
                                    global const signed char *sc2) {}

// CIR-LABEL: cir.func{{.*}} @signed_char_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-SAME: addr_space = [#cir<lang_address_space(default)>, #cir<lang_address_space(offload_global)>]
// CIR-SAME: access_qual = ["none", "none"]
// CIR-SAME: type = ["char", "char*"]
// CIR-SAME: base_type = ["char", "char*"]
// CIR-SAME: type_qual = ["", "const"]
// CIR-ARGINFO-LABEL: cir.func{{.*}} @signed_char_kernel_args
// CIR-ARGINFO-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata
// CIR-ARGINFO-SAME: addr_space = [#cir<lang_address_space(default)>, #cir<lang_address_space(offload_global)>]
// CIR-ARGINFO-SAME: access_qual = ["none", "none"]
// CIR-ARGINFO-SAME: type = ["char", "char*"]
// CIR-ARGINFO-SAME: base_type = ["char", "char*"]
// CIR-ARGINFO-SAME: type_qual = ["", "const"]
// CIR-ARGINFO-SAME: name = ["sc1", "sc2"]

// LLVM-DAG: define{{.*}} void @signed_char_kernel_args{{.+}} !kernel_arg_addr_space ![[SIGNED_CHAR_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS]] !kernel_arg_type ![[SIGNED_CHAR_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[SIGNED_CHAR_ARG_TYPES]] !kernel_arg_type_qual ![[SIGNED_CHAR_TYPE_QUALS:[0-9]+]]
// LLVM-ARGINFO-DAG: define{{.*}} void @signed_char_kernel_args{{.+}} !kernel_arg_addr_space ![[SIGNED_CHAR_ADDR_SPACES:[0-9]+]] !kernel_arg_access_qual ![[CONSTANT_ACCESS_QUALS]] !kernel_arg_type ![[SIGNED_CHAR_ARG_TYPES:[0-9]+]] !kernel_arg_base_type ![[SIGNED_CHAR_ARG_TYPES]] !kernel_arg_type_qual ![[SIGNED_CHAR_TYPE_QUALS:[0-9]+]] !kernel_arg_name ![[SIGNED_CHAR_ARG_NAMES:[0-9]+]]

// LLVM-DAG: ![[SIGNED_CHAR_ADDR_SPACES]] = !{i32 0, i32 1}
// LLVM-DAG: ![[SIGNED_CHAR_ARG_TYPES]] = !{!"char", !"char*"}
// LLVM-DAG: ![[SIGNED_CHAR_TYPE_QUALS]] = !{!"", !"const"}
// LLVM-ARGINFO-DAG: ![[SIGNED_CHAR_ARG_NAMES]] = !{!"sc1", !"sc2"}
