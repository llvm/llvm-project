# This test is to verify that BOLT won't take a label pointing to constant
# island as a secondary entry point. This could happen when function doesn't
# have ELF size set if it is from assembly code, or a constant island is
# referenced by another function discovered during relocation processing.

# RUN: split-file %s %t

# RUN: %clang %cflags -pie %t/tt.asm -o %t.so \
# RUN:   -Wl,-q -Wl,--init=_foo -Wl,--fini=_foo
# RUN: llvm-bolt %t.so -o %t.bolt.so --print-cfg 2>&1 | FileCheck %s
# CHECK-NOT: BOLT-WARNING: reference in the middle of instruction detected \
# CHECK-NOT:   function _start at offset 0x{{[0-9a-f]+}}
# CHECK: Binary Function "_start" after building cfg

# RUN: %clang %cflags -ffunction-sections -shared %t/tt.c %t/ss.c -o %tt.so \
# RUN:   -Wl,-q -Wl,--init=_start -Wl,--fini=_start \
# RUN:   -Wl,--version-script=%t/linker_script
# RUN: llvm-bolt %tt.so -o %tt.bolted.so

;--- tt.asm
  .text

  .global _foo
  .type _foo, %function
_foo:
  ret

  .global _start
  .type _start, %function
_start:
  b _foo

  .balign 16
_random_consts:
  .long 0x12345678
  .long 0x90abcdef

  .global _bar
  .type _bar, %function
_bar:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE

;--- tt.c
void _start() {}

__attribute__((naked)) void foo() {
  asm("ldr x16, .L_fnptr\n"
      "blr x16\n"
      "ret\n"

      "_rodatx:"
      ".global _rodatx;"
      ".quad 0;"
      ".L_fnptr:"
      ".quad 0;");
}

;--- ss.c
__attribute__((visibility("hidden"))) extern void* _rodatx;
void* bar() { return &_rodatx; }

;--- linker_script
{
global:
  _start;
  foo;
  bar;
local: *;
};
