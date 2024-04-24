# REQUIRES: x86
# RUN: split-file %s %t.dir

# RUN: llvm-dlltool -m i386:x86-64 -d %t.dir/lib.def -D lib.dll -l %t.dir/lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/main.s -filetype=obj -o %t.dir/main.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/func.s -filetype=obj -o %t.dir/func.obj
# RUN: env LLD_IN_TEST=1 not lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/lib.lib 2>&1 | FileCheck %s --check-prefix=ERR

# RUN: lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/func.obj %t.dir/lib.lib 2>&1 | FileCheck %s --check-prefix=NOERR --allow-empty

# ERR: error: output image has runtime pseudo relocations, but the function _pei386_runtime_relocator is missing; it is needed for fixing the relocations at runtime

# NOERR-NOT: error

#--- main.s
    .global main
    .text
main:
    ret

    .data
    .long 1
    .quad variable
    .long 2

#--- func.s
    .global _pei386_runtime_relocator
    .text
_pei386_runtime_relocator:
    ret

#--- lib.def
EXPORTS
variable DATA

