// This test is added to provide test coverage for the uwtable attribute. It
// aims to make sure that for an X86_64 output compiled with -fexceptions or
//  -fno-exceptions, a uwtable attribute is emitted. However, for an AArch64
//  output with -fexceptions a uwtable attribute is emitted, but with 
// -fno-exceptions a uwtable attribute is not emitted.

// RUN: %clang -g -fexceptions --target=x86_64-apple-darwin %s -S -emit-llvm -o - | FileCheck %s --check-prefix=X86_64_UWTABLE_EXCEPTIONS
// X86_64_UWTABLE_EXCEPTIONS: attributes #0 = { {{.*}} uwtable
// X86_64_UWTABLE_EXCEPTIONS: !{{[0-9]+}} = !{i32 7, !"uwtable", i32 2}

// RUN: %clang -g -fno-exceptions --target=x86_64-apple-darwin %s -S -emit-llvm -o - | FileCheck %s --check-prefix=X86_64_UWTABLE
// X86_64_UWTABLE: attributes #0 = { {{.*}} uwtable
// X86_64_UWTABLE: !{{[0-9]+}} = !{i32 7, !"uwtable", i32 2}

// RUN: %clang -g -fexceptions --target=arm64-apple-darwin %s -S -emit-llvm -o - | FileCheck %s --check-prefix=ARM64_UWTABLE_EXCEPTIONS
// ARM64_UWTABLE_EXCEPTIONS: attributes #0 = { {{.*}} uwtable
// ARM64_UWTABLE_EXCEPTIONS: !{{[0-9]+}} = !{i32 7, !"uwtable", i32 1}

// RUN: %clang -g -fno-exceptions --target=arm64-apple-darwin %s -S -emit-llvm -o - | FileCheck %s --check-prefix=ARM64_UWTABLE
// ARM64_UWTABLE-NOT: attributes #0 = { {{.*}} uwtable
// ARM64_UWTABLE-NOT: !{{[0-9]+}} = !{i32 7, !"uwtable", i32 2}
// ARM64_UWTABLE: attributes #0 =

int main() {
    return 1;
}
