# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef-debug.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef-bad-debug.s -o %t4.o
# RUN: rm -f %t2.a
# RUN: llvm-ar rc %t2.a %t2.o
# RUN: not ld.lld %t.o %t2.a %t3.o %t4.o -o /dev/null 2>&1 \
# RUN:   | FileCheck %s --implicit-check-not="error:" --implicit-check-not="warning:"
# RUN: not ld.lld -pie %t.o %t2.a %t3.o %t4.o -o /dev/null 2>&1 \
# RUN:   | FileCheck %s --implicit-check-not="error:" --implicit-check-not="warning:"

# CHECK:      error: undefined symbol: foo
# CHECK-NEXT: >>> referenced by undef.s
# CHECK-NEXT:                   {{.*}}:(.text+0x1)

# CHECK:      error: undefined symbol: bar
# CHECK-NEXT: >>> referenced by undef.s
# CHECK-NEXT: >>>               {{.*}}:(.text+0x6)

# CHECK:      error: undefined symbol: foo(int)
# CHECK-NEXT: >>> referenced by undef.s
# CHECK-NEXT: >>>               {{.*}}:(.text+0x10)

# CHECK:      error: undefined symbol: vtable for Foo 
# CHECK-NEXT: >>> referenced by undef.s
# CHECK-NEXT: >>>               {{.*}}:(.text+0x15)
# CHECK-NEXT: >>> the vtable symbol may be undefined because the class is missing its key function (see https://lld.llvm.org/missingkeyfunction)

# Check that this symbol isn't demangled

# CHECK:      error: undefined symbol: __Z3fooi
# CHECK-NEXT: >>> referenced by undef.s
# CHECK-NEXT: >>>               {{.*}}:(.text+0x1A)

# CHECK:      error: undefined symbol: zed2
# CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x0) in archive {{.*}}2.a

# CHECK:      error: undefined symbol: zed3
# CHECK-NEXT: >>> referenced by undef-debug.s:3 (dir{{/|\\}}undef-debug.s:3)
# CHECK-NEXT: >>>               {{.*}}.o:(.text+0x0)

# CHECK:      error: undefined symbol: zed4
# CHECK-NEXT: >>> referenced by undef-debug.s:7 (dir{{/|\\}}undef-debug.s:7)
# CHECK-NEXT: >>>               {{.*}}.o:(.text.1+0x0)

# CHECK:      error: undefined symbol: zed5
# CHECK-NEXT: >>> referenced by undef-debug.s:11 (dir{{/|\\}}undef-debug.s:11)
# CHECK-NEXT: >>>               {{.*}}.o:(.text.2+0x0)

# Show that all line table problems are mentioned as soon as the object's line information
# is requested, even if that particular part of the line information is not currently required.
# Also show that the warnings are only printed once.
# CHECK:      warning: unknown data in line table prologue at offset 0x00000000: parsing ended (at offset 0x00000037) before reaching the prologue at offset 0x00000038
# CHECK-NEXT: warning: parsing line table prologue at offset 0x0000005b: unsupported version 1
# CHECK-NEXT: warning: last sequence in debug line table at offset 0x00000061 is not terminated
# CHECK:      error: undefined symbol: zed6a
# CHECK-NEXT: >>> referenced by undef-bad-debug.s:11 (dir{{/|\\}}undef-bad-debug.s:11)
# CHECK-NEXT: >>>               {{.*}}4.o:(.text+0x0)
# CHECK:      error: undefined symbol: zed6b
# CHECK-NEXT: >>> referenced by undef-bad-debug.s:21 (dir{{/|\\}}undef-bad-debug.s:21)
# CHECK-NEXT: >>>               {{.*}}4.o:(.text+0x8)

# Show that a problem in a line table that prevents further parsing of that
# table means that no line information is displayed in the wardning.
# CHECK:      error: undefined symbol: zed7
# CHECK-NEXT: >>> referenced by {{.*}}4.o:(.text+0x10)

# Show that a problem with one line table's information doesn't affect getting information from
# a different one in the same object.
# CHECK:      error: undefined symbol: zed8
# CHECK-NEXT: >>> referenced by undef-bad-debug2.s:11 (dir2{{/|\\}}undef-bad-debug2.s:11)
# CHECK-NEXT: >>>               {{.*}}tmp4.o:(.text+0x18)

# RUN: not ld.lld %t.o %t2.a -o /dev/null -no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO-DEMANGLE %s
# NO-DEMANGLE: error: undefined symbol: _Z3fooi

.file "undef.s"

  .globl _start
_start:
  call foo
  call bar
  call zed1
  call _Z3fooi
  call _ZTV3Foo
  call __Z3fooi
