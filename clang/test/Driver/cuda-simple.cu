// Verify that we can parse a simple CUDA file with or without -save-temps
// http://llvm.org/PR22936
// RUN: %clang -nocudainc -nocudalib -Werror -fsyntax-only -c %s
//
// Verify that we pass -x cuda-cpp-output to compiler after 
// preprocessing a CUDA file
// RUN: %clang  -Werror -### -save-temps -c %s 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK: "-E"
// CHECK: "-x" "cuda"
// CHECK-NEXT: "-cc1"
// CHECK: "-x" "cuda-cpp-output"
//
// Verify that compiler accepts CUDA syntax with "-x cuda-cpp-output".
// RUN: %clang -Werror -fsyntax-only -x cuda-cpp-output -c %s

int cudaConfigureCall(int, int);
__attribute__((global)) void kernel() {}

void func() {
     kernel<<<1,1>>>();
}

