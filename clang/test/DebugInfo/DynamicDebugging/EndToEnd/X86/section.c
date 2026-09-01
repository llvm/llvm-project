// By default LLVM gives this section SHF_EXCLUDE, which we don't want. We
// expect a section named .debug_llvm_dyndbg of type LLVM_DYNDBG_ELF with
// no flags and alignment of 8 bytes.

// RUN: %clang -cc1 %s -emit-obj -debug-info-kind=limited -fdynamic-debugging -o - -triple x86_64-unknown-unknown | llvm-readelf --section-details - \
// RUN: | FileCheck %s
//             [Nr] Name
// CHECK:      .debug_llvm_dyndbg
//             Type            Address          Off           Size          ES Lk Inf Al
// CHECK-NEXT: LLVM_DYNDBG_ELF 0000000000000000 {{[0-9a-f]+}} {{[0-9a-f]+}} 00 0  0   8
//             Flags
// CHECK-NEXT: [0000000000000000]: {{$}}

int g;
int b() { return g; }
