// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++1.0 -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=clc++1.0 -o - %s -DWIN| FileCheck %s --check-prefixes=WINDOWS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=CL2.0 -o - %s -DWIN| FileCheck %s --check-prefixes=OCLWINDOWS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space -o - %s -DWIN | FileCheck %s --check-prefixes=OCLWINDOWS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables,-__opencl_c_device_enqueue -o - %s -DWIN | FileCheck %s --check-prefixes=OCLWINDOWS

typedef unsigned char __attribute__((ext_vector_type(3))) uchar3;
typedef int __attribute__((ext_vector_type(4))) int4;

__attribute__((overloadable))
void test1(read_only pipe int p) {
// LINUX: define{{.*}} void @_Z5test18ocl_pipe
// WINDOWS: define dso_local void @"?test1@@YAXU?$ocl_pipe@H$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test1@@$$J0YAXU?$ocl_pipe@H$00@__clang@@@Z"
}

__attribute__((overloadable))
void test2(write_only pipe float p) {
// LINUX: define{{.*}} void @_Z5test28ocl_pipe
// WINDOWS: define dso_local void @"?test2@@YAXU?$ocl_pipe@M$0A@@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test2@@$$J0YAXU?$ocl_pipe@M$0A@@__clang@@@Z"
}

#ifdef WIN
// It isn't possible to overload on pipe types in Linux mode
// because the OCL specification on the Itanium ABI has a specified mangling
// for the entire class of types, and thus doesn't take element type or read/write
// into account. Thus, both would result in the same mangling, which is an IR-CodeGen
// error. Our windows implementation of this mangling doesn't have that problem,
// so we can test it here.
__attribute__((overloadable))
void test2(read_only pipe int p) {
// WINDOWS: define dso_local void @"?test2@@YAXU?$ocl_pipe@H$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test2@@$$J0YAXU?$ocl_pipe@H$00@__clang@@@Z"
}
#endif

__attribute__((overloadable))
void test3(read_only pipe const int p) {
// LINUX: define{{.*}} void @_Z5test38ocl_pipe
// WINDOWS: define dso_local void @"?test3@@YAXU?$ocl_pipe@$$CBH$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test3@@$$J0YAXU?$ocl_pipe@$$CBH$00@__clang@@@Z"
}

__attribute__((overloadable))
void test4(read_only pipe uchar3 p) {
// LINUX: define{{.*}} void @_Z5test48ocl_pipe
// WINDOWS: define dso_local void @"?test4@@YAXU?$ocl_pipe@T?$__vector@E$02@__clang@@$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test4@@$$J0YAXU?$ocl_pipe@T?$__vector@E$02@__clang@@$00@__clang@@@Z"
}

__attribute__((overloadable))
void test5(read_only pipe int4 p) {
// LINUX: define{{.*}} void @_Z5test58ocl_pipe
// WINDOWS: define dso_local void @"?test5@@YAXU?$ocl_pipe@T?$__vector@H$03@__clang@@$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test5@@$$J0YAXU?$ocl_pipe@T?$__vector@H$03@__clang@@$00@__clang@@@Z"
}

typedef read_only pipe int MyPipe;
kernel void test6(MyPipe p) {
// LINUX: define{{.*}} spir_kernel void @test6
// WINDOWS: define dso_local spir_kernel void @test6
// OCLWINDOWS: define dso_local spir_kernel void @test6
}

struct Person {
  const char *Name;
  bool isFemale;
  int ID;
};

__attribute__((overloadable))
void test_reserved_read_pipe(global struct Person *SDst,
                             read_only pipe struct Person SPipe) {
// LINUX: define{{.*}} void @_Z23test_reserved_read_pipePU8CLglobal6Person8ocl_pipe
// WINDOWS: define dso_local void @"?test_reserved_read_pipe@@YAXPEAU?$_ASCLglobal@$$CAUPerson@@@__clang@@U?$ocl_pipe@UPerson@@$00@2@@Z"
// OCLWINDOWS: define dso_local void @"?test_reserved_read_pipe@@$$J0YAXPEAU?$_ASCLglobal@$$CAUPerson@@@__clang@@U?$ocl_pipe@UPerson@@$00@2@@Z"
}
