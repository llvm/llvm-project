## TODO

# RUN: %clang %cflags %s -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg --print-only=foo 2>&1 | FileCheck %s

# BOLT-WARNING: function symbol foo lacks code marker

.text
.balign 4

.global _start
.type _start, %function
_start:
  bl foo
  ret
.size _start, .-_start

## Data marker is emitted immediately before the function.
.global foo
.type foo, %function
$d:
foo:
  ret
.size foo, .-foo

