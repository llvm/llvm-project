# REQUIRES: x86
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: not ld.lld a.o a.o 2>&1 | FileCheck --check-prefix=DEMANGLE %s --implicit-check-not=error:

# DEMANGLE:      error: duplicate symbol: mul(double, double)
# DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# DEMANGLE:      error: duplicate symbol: foo
# DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)

# RUN: not ld.lld a.o a.o --no-demangle 2>&1 | FileCheck --check-prefix=NO_DEMANGLE %s --implicit-check-not=error:

# NO_DEMANGLE:      error: duplicate symbol: _Z3muldd
# NO_DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# NO_DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# NO_DEMANGLE:      error: duplicate symbol: foo
# NO_DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# NO_DEMANGLE-NEXT: >>> defined at {{.*}}:(.text+0x0)

# RUN: not ld.lld a.o a.o --demangle --no-demangle 2>&1 | FileCheck --check-prefix=NO_DEMANGLE %s --implicit-check-not=error:
# RUN: not ld.lld a.o a.o --no-demangle --demangle 2>&1 | FileCheck --check-prefix=DEMANGLE %s --implicit-check-not=error:

# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/conflict.s -o b.o
# RUN: rm -f b.a
# RUN: llvm-ar rcs b.a b.o
# RUN: not ld.lld a.o b.a -u baz 2>&1 | FileCheck --check-prefix=ARCHIVE %s --implicit-check-not=error:

# ARCHIVE:      error: duplicate symbol: mul(double, double)
# ARCHIVE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# ARCHIVE-NEXT: >>> defined at {{.*}}:(.text+0x0) in archive {{.*}}.a
# ARCHIVE:      error: duplicate symbol: foo
# ARCHIVE-NEXT: >>> defined at {{.*}}:(.text+0x0)
# ARCHIVE-NEXT: >>> defined at {{.*}}:(.text+0x0) in archive {{.*}}.a

# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/conflict-debug.s -o dbg.o
# RUN: not ld.lld dbg.o dbg.o 2>&1 | FileCheck --check-prefix=DBGINFO %s --implicit-check-not=error:

# DBGINFO:      error: duplicate symbol: zed
# DBGINFO-NEXT: >>> defined at conflict-debug.s:4
# DBGINFO-NEXT: >>>            {{.*}}:(.text+0x0)
# DBGINFO-NEXT: >>> defined at conflict-debug.s:4
# DBGINFO-NEXT: >>>            {{.*}}:(.text+0x0)

.globl _Z3muldd, foo
_Z3muldd:
foo:
  mov $60, %rax
  mov $42, %rdi
  syscall
