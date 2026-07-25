// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -mrelocation-model static -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -mrelocation-model static -O2 -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -mrelocation-model static -O2 -emit-obj -o %t.o %s
// RUN: obj2yaml %t.o | FileCheck %s --check-prefix=OBJ

// IR: @mut = {{.*}}addrspace(1) global i32 33554435
int mut __attribute__((address_space(1))) = 0x02000003;

// Without volatile the load of a constant global folds to i32.const, hiding
// the global.get.
// IR: @immut = {{.*}}addrspace(1) constant i32 42
const volatile int immut __attribute__((address_space(1))) = 42;

// ASM-LABEL: read_mut:
// ASM: global.get mut{{$}}
int read_mut(void) { return mut; }

// ASM-LABEL: write_mut:
// ASM: global.set mut{{$}}
void write_mut(int x) { mut = x; }

// ASM-LABEL: read_immut:
// ASM: global.get immut{{$}}
int read_immut(void) { return immut; }

// ASM: .globaltype mut, i32{{$}}
// ASM: .globaltype immut, i32, immutable{{$}}

// OBJ:      - Type:            GLOBAL
// OBJ-NEXT:   Globals:
// OBJ-NEXT:     - Index:           0
// OBJ-NEXT:       Type:            I32
// OBJ-NEXT:       Mutable:         true
// OBJ-NEXT:       InitExpr:
// OBJ-NEXT:         Opcode:          I32_CONST
// OBJ-NEXT:         Value:           33554435
// OBJ-NEXT:     - Index:           1
// OBJ-NEXT:       Type:            I32
// OBJ-NEXT:       Mutable:         false
// OBJ-NEXT:       InitExpr:
// OBJ-NEXT:         Opcode:          I32_CONST
// OBJ-NEXT:         Value:           42
