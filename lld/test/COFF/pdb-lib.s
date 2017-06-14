# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=i686-windows-msvc %s -o foo.obj
# RUN: llc %S/Inputs/bar.ll -filetype=obj -mtriple=i686-windows-msvc -o bar.obj
# RUN: llvm-lib bar.obj -out:bar.lib
# RUN: lld-link -debug -pdb:foo.pdb foo.obj bar.lib -out:foo.exe -entry:main
# RUN: llvm-pdbutil raw -modules %t/foo.pdb | FileCheck %s

# Make sure that the PDB has module descriptors. foo.obj and bar.lib should be
# absolute paths, and bar.obj should be the relative path passed to llvm-lib.

# CHECK-LABEL: Modules [
# CHECK-NEXT:   {
# CHECK-NEXT:     Name: {{.*pdb-lib.s.tmp[/\\]foo.obj}}
# CHECK-NEXT:     Debug Stream Index:
# CHECK-NEXT:     Object File Name: {{.*pdb-lib.s.tmp[/\\]foo.obj}}
# CHECK-NEXT:     Num Files: 0
# CHECK-NEXT:     Source File Name Idx: 0
# CHECK-NEXT:     Pdb File Name Idx: 0
# CHECK-NEXT:     Line Info Byte Size: 0
# CHECK-NEXT:     C13 Line Info Byte Size: 0
# CHECK-NEXT:     Symbol Byte Size: 4
# CHECK-NEXT:     Type Server Index: 0
# CHECK-NEXT:     Has EC Info: No
# CHECK-NEXT:   }
# CHECK-NEXT:   {
# CHECK-NEXT:     Name: bar.obj
# CHECK-NEXT:     Debug Stream Index:
# CHECK-NEXT:     Object File Name: {{.*pdb-lib.s.tmp[/\\]bar.lib}}
# CHECK-NEXT:     Num Files: 0
# CHECK-NEXT:     Source File Name Idx: 0
# CHECK-NEXT:     Pdb File Name Idx: 0
# CHECK-NEXT:     Line Info Byte Size: 0
# CHECK-NEXT:     C13 Line Info Byte Size: 0
# CHECK-NEXT:     Symbol Byte Size: 4
# CHECK-NEXT:     Type Server Index: 0
# CHECK-NEXT:     Has EC Info: No
# CHECK-NEXT:   }
# CHECK-NEXT:   {
# CHECK-NEXT:     Name: * Linker *
# CHECK-NEXT:     Debug Stream Index:
# CHECK-NEXT:     Object File Name:
# CHECK-NEXT:     Num Files: 0
# CHECK-NEXT:     Source File Name Idx: 0
# CHECK-NEXT:     Pdb File Name Idx: 0
# CHECK-NEXT:     Line Info Byte Size: 0
# CHECK-NEXT:     C13 Line Info Byte Size: 0
# CHECK-NEXT:     Symbol Byte Size: 4
# CHECK-NEXT:     Type Server Index: 0
# CHECK-NEXT:     Has EC Info: No
# CHECK-NEXT:   }
# CHECK-NEXT: ]


        .def     _main;
        .scl    2;
        .type   32;
        .endef
        .globl  _main
_main:
        calll _bar
        xor %eax, %eax
        retl

