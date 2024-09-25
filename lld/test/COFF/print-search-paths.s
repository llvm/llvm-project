# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -safeseh:no /dll /noentry /winsysroot:%t.dir/sysroot /vctoolsversion:1.1.1.1 /winsdkversion:10.0.1 /libpath:custom-dir %t.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir %s
# RUN: llvm-mc -triple i686-windows-msvc %s -filetype=obj -o %t_32.obj
# RUN: lld-link -safeseh:no /dll /noentry /winsysroot:%t.dir/sysroot /vctoolsversion:1.1.1.1 /winsdkversion:10.0.1 %t_32.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir -check-prefix=X86 %s
# CHECK: Library search paths:
# CHECK-NEXT:   (cwd)
# CHECK-NEXT:   custom-dir
# CHECK-NEXT:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# CHECK-NEXT:   [[CPATH]]lib
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}DIA SDK{{[/\\]}}lib{{[/\\]}}amd64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}x64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}x64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}x64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}x64
# X86: Library search paths:
# X86-NEXT:   (cwd)
# X86-NEXT:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# X86-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# X86-NEXT:   [[CPATH]]lib
# X86-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}DIA SDK{{[/\\]}}lib
# X86-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}x86
# X86-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}x86
# X86-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}x86
# X86-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}x86


        .text
        .globl  _main
_main:
        ret
