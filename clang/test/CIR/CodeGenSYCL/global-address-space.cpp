// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck --input-file=%t.pre.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// SYCL device globals, static locals and string literals with no explicit
// address space are emitted in the global address space, matching classic
// CodeGen. Their uses are cast to the generic address space. Globals with an
// explicit address space keep it.

int g = 1;
__attribute__((opencl_constant)) int explicit_c = 2;
const char *gs = "Hello, world!";
int i;

// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @g = #cir.int<1> : !s32i
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_constant) @explicit_c = #cir.int<2> : !s32i
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @".str" = #cir.const_array<"Hello, world!" : !cir.array<!s8i x 13>, trailing_zeros> : !cir.array<!s8i x 14>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @gs = #cir.global_view<@".str"> : !cir.ptr<!s8i, target_address_space(4)>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @i = #cir.int<0> : !s32i
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @_ZZ18static_const_localvE3foo = #cir.int<66> : !s32i
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @".str.1" = #cir.const_array<"Another hello world!" : !cir.array<!s8i x 20>, trailing_zeros> : !cir.array<!s8i x 21>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @".str.2" = #cir.const_array<"Yet another Hello world" : !cir.array<!s8i x 23>, trailing_zeros> : !cir.array<!s8i x 24>

// LLVM-DAG: @g = addrspace(1) global i32 1, align 4
// LLVM-DAG: @explicit_c = addrspace(2) constant i32 2, align 4
// LLVM-DAG: @gs = addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), align 8
// LLVM-DAG: @i = addrspace(1) global i32 0, align 4
// LLVM-DAG: @_ZZ18static_const_localvE3foo = internal addrspace(1) constant i32 66, align 4
// LLVM-CIR-DAG: @.str = private addrspace(1) constant [14 x i8] c"Hello, world!\00", align 1
// LLVM-CIR-DAG: @.str.1 = private addrspace(1) constant [21 x i8] c"Another hello world!\00", align 1
// LLVM-CIR-DAG: @.str.2 = private addrspace(1) constant [24 x i8] c"Yet another Hello world\00", align 1
// OGCG-DAG: @.str = private unnamed_addr addrspace(1) constant [14 x i8] c"Hello, world!\00", align 1
// OGCG-DAG: @.str.1 = private unnamed_addr addrspace(1) constant [21 x i8] c"Another hello world!\00", align 1
// OGCG-DAG: @.str.2 = private unnamed_addr addrspace(1) constant [24 x i8] c"Yet another Hello world\00", align 1

[[clang::sycl_external]] int *addr_g() { return &g; }

// CIR-LABEL: cir.func {{.*}} @_Z6addr_gv()
// CIR:         %[[G:.*]] = cir.get_global @g : !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR:         cir.cast address_space %[[G]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!s32i, target_address_space(4)>

// LLVM-LABEL: define {{.*}} ptr addrspace(4) @_Z6addr_gv()
// LLVM:         addrspacecast (ptr addrspace(1) @g to ptr addrspace(4))

[[clang::sycl_external]] __attribute__((opencl_constant)) int *addr_explicit_c() {
  return &explicit_c;
}

// CIR-LABEL: cir.func {{.*}} @_Z15addr_explicit_cv()
// CIR:         cir.get_global @explicit_c : !cir.ptr<!s32i, lang_address_space(offload_constant)>
// CIR-NOT:     cir.cast address_space
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} ptr addrspace(2) @_Z15addr_explicit_cv()
// LLVM-NOT:     addrspacecast
// LLVM:         ret ptr addrspace(2)

[[clang::sycl_external]] const int *static_const_local() {
  static const int foo = 0x42;
  return &foo;
}

// CIR-LABEL: cir.func {{.*}} @_Z18static_const_localv()
// CIR:         %[[FOO:.*]] = cir.get_global @_ZZ18static_const_localvE3foo : !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR:         cir.cast address_space %[[FOO]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!s32i, target_address_space(4)>

// LLVM-LABEL: define {{.*}} ptr addrspace(4) @_Z18static_const_localv()
// LLVM:         addrspacecast (ptr addrspace(1) @_ZZ18static_const_localvE3foo to ptr addrspace(4))

[[clang::sycl_external]] const char *str() { return "Hello, world!"; }

// CIR-LABEL: cir.func {{.*}} @_Z3strv()
// CIR:         %[[STR:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 14>, lang_address_space(offload_global)>
// CIR:         cir.cast address_space %[[STR]] : !cir.ptr<!cir.array<!s8i x 14>, lang_address_space(offload_global)> -> !cir.ptr<!cir.array<!s8i x 14>, target_address_space(4)>

// LLVM-LABEL: define {{.*}} ptr addrspace(4) @_Z3strv()
// LLVM:         addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4))

[[clang::sycl_external]] const char *phi_str() {
  return i > 2 ? gs : "Another hello world!";
}

// LLVM-LABEL: define {{.*}} ptr addrspace(4) @_Z7phi_strv()
// LLVM:         load ptr addrspace(4), ptr addrspace(4) addrspacecast (ptr addrspace(1) @gs to ptr addrspace(4)), align 8
// LLVM:         phi ptr addrspace(4) {{.*}}addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4))

[[clang::sycl_external]] const char *select_null() {
  return i > 2 ? "Yet another Hello world" : nullptr;
}

// LLVM-LABEL: define {{.*}} ptr addrspace(4) @_Z11select_nullv()
// LLVM:         select i1 %{{.*}}, ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str.2 to ptr addrspace(4)), ptr addrspace(4) null
