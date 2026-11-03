# REQUIRES: arm

# RUN: llvm-mc -triple arm-windows-msvc %s -filetype=obj -o %t.arm.obj
# RUN: lld-link -dll -noentry -winsysroot:%t.dir/sysroot -vctoolsversion:1.1.1.1 -winsdkversion:10.0.1 -libpath:custom-dir \
# RUN:           %t.arm.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir %s --check-prefixes=CHECK

# CHECK: Library search paths:
# CHECK-NEXT:   (cwd)
# CHECK-NEXT:   custom-dir
# CHECK-NEXT:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# CHECK-NEXT:   [[CPATH]]lib
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}armv7-pc-windows-msvc
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}arm-pc-windows-msvc
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}thumbv7-pc-windows-msvc
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}DIA SDK{{[/\\]}}lib{{[/\\]}}arm
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}arm
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}arm
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}arm
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}arm

        .data
        .word 1
