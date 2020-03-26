// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++ -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=clc++ -o - %s -DWIN| FileCheck %s --check-prefixes=WINDOWS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s --check-prefixes=UNMANGLED,OCLLINUX
// RUN: %clang_cc1 -triple x86_64-unknown-windows-pc -emit-llvm -O0 -cl-std=CL2.0 -o - %s -DWIN| FileCheck %s --check-prefixes=UNMANGLED,OCLWINDOWS

typedef unsigned char __attribute__((ext_vector_type(3))) uchar3;
typedef int __attribute__((ext_vector_type(4))) int4;

void test1(read_only pipe int p) {
// LINUX: define void @_Z5test18ocl_pipe
// WINDOWS: define dso_local void @"?test1@@YAXU?$ocl_pipe@H$00@__clang@@@Z"
// UNMANGLED: define {{.*}}void @test1(
}

__attribute__((overloadable))
void test2(write_only pipe float p) {
// LINUX: define void @_Z5test28ocl_pipe
// WINDOWS: define dso_local void @"?test2@@YAXU?$ocl_pipe@M$0A@@__clang@@@Z"
// Note: overloadable attribute makes OpenCL Linux still mangle this,
// but we cannot overload on pipe still.
// OCLLINUX: define void @_Z5test28ocl_pipe
// OCLWINDOWS: define dso_local void @"?test2@@$$J0YAXU?$ocl_pipe@M$0A@@__clang@@@Z"
}

#ifdef WIN
// SPIR Spec specifies mangling on pipes that doesn't include the element type
//  or write/read. Our Windows mangling does, so make sure this still works.
__attribute__((overloadable))
void test2(read_only pipe int p) {
// WINDOWS: define dso_local void @"?test2@@YAXU?$ocl_pipe@H$00@__clang@@@Z"
// OCLWINDOWS: define dso_local void @"?test2@@$$J0YAXU?$ocl_pipe@H$00@__clang@@@Z"
}
#endif


void test3(read_only pipe const int p) {
// LINUX: define void @_Z5test38ocl_pipe
// WINDOWS: define dso_local void @"?test3@@YAXU?$ocl_pipe@$$CBH$00@__clang@@@Z"
// UNMANGLED: define {{.*}}void @test3(
}

void test4(read_only pipe uchar3 p) {
// LINUX: define void @_Z5test48ocl_pipe
// WINDOWS: define dso_local void @"?test4@@YAXU?$ocl_pipe@T?$__vector@E$02@__clang@@$00@__clang@@@Z"
// UNMANGLED: define {{.*}}void @test4(
}

void test5(read_only pipe int4 p) {
// LINUX: define void @_Z5test58ocl_pipe
// WINDOWS: define dso_local void @"?test5@@YAXU?$ocl_pipe@T?$__vector@H$03@__clang@@$00@__clang@@@Z"
// UNMANGLED: define {{.*}}void @test5(
}

typedef read_only pipe int MyPipe;
kernel void test6(MyPipe p) {
// LINUX: define spir_kernel void @test6
// WINDOWS: define dso_local spir_kernel void @test6
// UNMANGLED: define {{.*}}void @test6(
}

struct Person {
  const char *Name;
  bool isFemale;
  int ID;
};

void test_reserved_read_pipe(global struct Person *SDst,
                             read_only pipe struct Person SPipe) {
// LINUX: define void @_Z23test_reserved_read_pipePU8CLglobal6Person8ocl_pipe
// WINDOWS: define dso_local void @"?test_reserved_read_pipe@@YAXPEAU?$_ASCLglobal@$$CAUPerson@@@__clang@@U?$ocl_pipe@UPerson@@$00@2@@Z"
// UNMANGLED: define {{.*}}void @test_reserved_read_pipe(
}
