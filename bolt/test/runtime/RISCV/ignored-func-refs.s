## Exercise the same references as RISCV/ignored-func-refs.s. The program checks
## PC-relative/absolute addresses, alternate return registers and interior calls.
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj %S/../../RISCV/Inputs/ignored-func-refs.s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e _start %t.o -o %t.exe
# RUN: %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start -o %t.skip
# RUN: %t.skip
## Nearby AUIPC/JALR patches must not be relaxed and change return addresses.
# RUN: llvm-bolt %t.exe --skip-funcs=source,middle,_start --no-huge-pages --align-text=4 --align-functions=4 -o %t.near
# RUN: %t.near
# RUN: llvm-bolt %t.exe --funcs=short_target --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x11000 -o %t.short
# RUN: %t.short
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start --custom-allocation-vma=0x90000000 -o %t.far
# RUN: %t.far
# RUN: llvm-bolt %t.exe --force-patch -o %t.patch
# RUN: %t.patch
