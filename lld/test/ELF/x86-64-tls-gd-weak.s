// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: ld.lld -shared -Bsymbolic %t.o -o %t.so
// RUN: llvm-readobj -r %t.so | FileCheck %s

        .byte   0x66
        leaq    foo@tlsgd(%rip), %rdi
        .value  0x6666
        rex64
        call    __tls_get_addr@PLT

        .byte   0x66
        leaq    bar@tlsgd(%rip), %rdi
        .value  0x6666
        rex64
        call    __tls_get_addr@PLT

        .section        .tbss,"awT",@nobits
        .weak   foo
foo:
        .zero   4

        .hidden bar
        .weak   bar
bar:
        .zero   4

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
//
// Hidden weak TLS variables are non-preemptible. Only emit DTPMOD
// to resolve the runtime module ID; the offset is a link-time constant.
// CHECK-NEXT:     R_X86_64_DTPMOD64 - 0x0
//
// Global weak TLS variables are preemptible. Emit both DTPMOD and
// DTPOFF to resolve them together to the prevailing definition at runtime.
// CHECK-NEXT:     R_X86_64_DTPMOD64 foo 0x0
// CHECK-NEXT:     R_X86_64_DTPOFF64 foo 0x0
// CHECK-NEXT:   }
