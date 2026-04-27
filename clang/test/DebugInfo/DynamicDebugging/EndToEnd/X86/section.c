// By default LLVM gives this section SHF_EXCLUDE, which we don't want.

// RUN: %clang -cc1 %s -emit-obj -debug-info-kind=limited -fdynamic-debugging -o - -triple x86_64-unknown-unknown | llvm-readelf --section-details - \
// RUN: | FileCheck %s
//             [Nr] Name
// CHECK:      .debug_llvm_dyndbg
//             Type     Address          Off           Size          ES Lk Inf Al
// CHECK-NEXT: PROGBITS 0000000000000000 {{[0-9a-z]+}} {{[0-9a-z]+}} 00 0  0   1
//             Flags
// CHECK-NEXT: [0000000000000000]: {{$}}

int g;
int b() { return g; }
