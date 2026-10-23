## Share the executable input with the runtime test. Calls and address pairs
## are patched in place; references without sufficient reach retain targets.
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj %S/Inputs/ignored-func-refs.s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start -o %t.skip
# RUN: llvm-objdump -d %t.skip | FileCheck %s --check-prefix=PATCHED
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start --no-huge-pages --align-text=4 --align-functions=4 -o %t.near
# RUN: llvm-objdump -d %t.near | FileCheck %s --check-prefix=NEAR
# RUN: llvm-bolt %t.exe --funcs=short_target --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x11000 -o %t.short
# RUN: llvm-objdump -d %t.short | FileCheck %s --check-prefix=SHORT
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start --custom-allocation-vma=0x90000000 -o %t.far
# RUN: llvm-nm -n %t.far | FileCheck %s --check-prefix=FAR
# RUN: llvm-bolt %t.exe --force-patch -o %t.patch 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: llvm-objdump -d %t.patch | FileCheck %s --check-prefix=ENTRY

## Keep both instructions and the original return site, even for t0 calls.
# PATCHED:      1000c: {{.*}} auipc t0, 0x3f0
# PATCHED-NEXT: 10010: {{.*}} jalr t0, {{.*}} <leaf>
# PATCHED:      10040: {{.*}} jalr t0, {{.*}} <target_entry>
# PATCHED:      0000000000400000 <leaf>:

## A JAL can be patched when the new code is within its original reach.
# NEAR-LABEL: 0000000000010054 <source>:
# NEAR-NEXT: {{.*}} jal t0, 0x12{{[0-9a-f]+}} <middle>

## A compressed branch is patched without changing the following instruction.
# SHORT:      10fc4: {{.*}} beqz a0, 0x11000 <short_target>
# SHORT-NEXT: 10fc6: {{.*}} ret

## Beyond HI20/LO12 reach, preserve the targets instead of emitting a bad fixup.
# FAR: 0000000000010000 T leaf
# FAR: 0000000000010008 T middle
# FAR: 0000000000010048 T target
# FAR: 0000000000010fc0 T short_target

## Retaining source must cancel the entry patch already planned for middle.
# WARNING: BOLT-WARNING: failed to patch entries in source
## The eight-byte leaf entry fits a tail trampoline that preserves t0.
# ENTRY-LABEL: <leaf.org.0>:
# ENTRY-NEXT: {{.*}} auipc t1,
# ENTRY-NEXT: {{.*}} jr t1 {{.*}}<leaf>
# ENTRY-LABEL: 0000000000010008 <middle>:
# ENTRY-NEXT: {{.*}} mv s0, t0

## _end moves independently of a function that happens to share its address.
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e _start -u end_refs %t.o -o %t.end
# RUN: llvm-bolt %t.end --print-cfg --print-only=end_refs -o %t.end.bolt | FileCheck %s --check-prefix=END
# RUN: not llvm-bolt %t.end --skip-funcs=end_refs -o %t.error 2>&1 | FileCheck %s --check-prefix=ERROR
# END: auipc a1, %pcrel_hi(_end)
# ERROR: BOLT-ERROR: cannot update RISC-V instruction reference to _end at address 0x{{[0-9a-f]+}} in ignored function end_refs
