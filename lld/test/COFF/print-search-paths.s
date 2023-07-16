# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -safeseh:no /dll /noentry /winsysroot:%t.dir/sysroot /vctoolsversion:1.1.1.1 /winsdkversion:10.0.1 %t.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir %s
# RUN: llvm-mc -triple i686-windows-msvc %s -filetype=obj -o %t_32.obj
# RUN: lld-link -safeseh:no /dll /noentry /winsysroot:%t.dir/sysroot /vctoolsversion:1.1.1.1 /winsdkversion:10.0.1 %t_32.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir -check-prefix=X86 %s
# CHECK: Library search paths:
# CHECK:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# CHECK:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# CHECK:   [[CPATH]]lib
# CHECK:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}x64
# CHECK:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}x64
# CHECK:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}x64
# CHECK:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}x64
# X86: Library search paths:
# X86:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# X86:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# X86:   [[CPATH]]lib
# X86:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}x86
# X86:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}x86
# X86:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}x86
# X86:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}x86


        .text
        .globl  _main
_main:
        ret
