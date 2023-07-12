# REQUIRES: x86

# RUN: echo -e "EXPORTS\nvariable" > %t-lib.def
# RUN: llvm-dlltool -m i386 -d %t-lib.def -D lib.dll -l %t-lib.lib

# RUN: llvm-mc -triple=i386-windows-gnu %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw -out:%t.exe -entry:main %t.obj %t-lib.lib -verbose 2>&1 | FileCheck --allow-empty %s

# CHECK-NOT: runtime pseudo relocation {{.*}} is too narrow

    .global _main
    .text
_main:
    movl _variable, %eax
    ret

relocs:
    .long ___RUNTIME_PSEUDO_RELOC_LIST__
    .long ___RUNTIME_PSEUDO_RELOC_LIST_END__
