# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: not ld.lld %t1.o %t1.o -o %t2 2>&1 | FileCheck -check-prefix=DEMANGLE %s

# DEMANGLE:       {{.*}}:(.text+0x0): duplicate symbol 'mul(double, double)'
# DEMANGLE-NEXT:  {{.*}}:(.text+0x0): previous definition was here
# DEMANGLE-NEXT:  {{.*}}:(.text+0x0): duplicate symbol 'foo'
# DEMANGLE-NEXT:  {{.*}}:(.text+0x0): previous definition was here

# RUN: not ld.lld %t1.o %t1.o -o %t2 --no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO_DEMANGLE %s

# NO_DEMANGLE: {{.*}}:(.text+0x0): duplicate symbol '_Z3muldd'
# NO_DEMANGLE-NEXT: {{.*}}:(.text+0x0): previous definition was here
# NO_DEMANGLE-NEXT: {{.*}}:(.text+0x0): duplicate symbol 'foo'
# NO_DEMANGLE-NEXT: {{.*}}:(.text+0x0): previous definition was here

# RUN: not ld.lld %t1.o %t1.o -o %t2 --demangle --no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO_DEMANGLE %s
# RUN: not ld.lld %t1.o %t1.o -o %t2 --no-demangle --demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=DEMANGLE %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %S/Inputs/conflict.s -o %t2.o
# RUN: llvm-ar rcs %t3.a %t2.o
# RUN: not ld.lld %t1.o %t3.a -u baz -o %t2 2>&1 | FileCheck -check-prefix=ARCHIVE %s

# ARCHIVE:      {{.*}}3.a({{.*}}2.o):(.text+0x0): duplicate symbol 'foo'
# ARCHIVE-NEXT: {{.*}}1.o:(.text+0x0): previous definition was here

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/conflict-debug.s -o %t-dbg.o
# RUN: not ld.lld %t-dbg.o %t-dbg.o -o %t-dbg 2>&1 | FileCheck -check-prefix=DBGINFO %s

# DBGINFO:      conflict-debug.s:4: duplicate symbol 'zed'
# DBGINFO-NEXT: conflict-debug.s:4: previous definition was here

.globl _Z3muldd, foo
_Z3muldd:
foo:
  mov $60, %rax
  mov $42, %rdi
  syscall
