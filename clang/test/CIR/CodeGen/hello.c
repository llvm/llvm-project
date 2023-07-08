// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
int printf(const char *restrict, ...);

int main (void) {
    printf ("Hello, world!\n");
    return 0;
}

// CHECK: cir.func private @printf(!cir.ptr<!s8i>, ...) -> !s32i
// CHECK: cir.global "private" constant internal @".str" = #cir.const_array<"Hello, world!\0A\00" : !cir.array<!s8i x 15>> : !cir.array<!s8i x 15> {alignment = 1 : i64}
// CHECK: cir.func @main() -> !s32i
// CHECK:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.get_global @printf : cir.ptr <!cir.func<!s32i (!cir.ptr<!s8i>, ...)>>
// CHECK:   %2 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 15>>
// CHECK:   %3 = cir.cast(array_to_ptrdecay, %2 : !cir.ptr<!cir.array<!s8i x 15>>), !cir.ptr<!s8i>
// CHECK:   %4 = cir.call @printf(%3) : (!cir.ptr<!s8i>) -> !s32i
// CHECK:   %5 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:   cir.store %5, %0 : !s32i, cir.ptr <!s32i>
// CHECK:   %6 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:   cir.return %6 : !s32i
// CHECK: }
