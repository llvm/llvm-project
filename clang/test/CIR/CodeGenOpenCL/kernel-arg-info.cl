// See also clang/test/CodeGenOpenCL/kernel-arg-info.cl.
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple spirv64-unknown-unknown -emit-cir -cl-kernel-arg-info -o %t.arginfo.cir
// RUN: FileCheck %s --input-file=%t.arginfo.cir --check-prefix=CIR-ARGINFO

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
