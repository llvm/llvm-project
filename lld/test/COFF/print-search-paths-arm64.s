# REQUIRES: aarch64

# RUN: llvm-mc -triple aarch64-windows-msvc %s -filetype=obj -o %t.aarch64.obj
# RUN: lld-link -dll -noentry -winsysroot:%t.dir/sysroot -vctoolsversion:1.1.1.1 -winsdkversion:10.0.1 -libpath:custom-dir \
# RUN:           %t.aarch64.obj -print-search-paths | FileCheck -DSYSROOT=%t.dir %s

# RUN: llvm-mc -triple arm64ec-windows-msvc %s -filetype=obj -o %t.arm64ec.obj
# RUN: lld-link -dll -noentry -winsysroot:%t.dir/sysroot -vctoolsversion:1.1.1.1 -winsdkversion:10.0.1 -libpath:custom-dir \
# RUN:           %t.arm64ec.obj -print-search-paths -machine:arm64ec | FileCheck -DSYSROOT=%t.dir %s

# CHECK: Library search paths:
# CHECK-NEXT:   (cwd)
# CHECK-NEXT:   custom-dir
# CHECK-NEXT:   [[CPATH:.*]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib{{[/\\]}}windows
# CHECK-NEXT:   [[CPATH]]lib{{[/\\]}}clang{{[/\\]}}{{[0-9]+}}{{[/\\]}}lib
# CHECK-NEXT:   [[CPATH]]lib
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}DIA SDK{{[/\\]}}lib{{[/\\]}}arm64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}lib{{[/\\]}}arm64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}VC{{[/\\]}}Tools{{[/\\]}}MSVC{{[/\\]}}1.1.1.1{{[/\\]}}atlmfc{{[/\\]}}lib{{[/\\]}}arm64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}ucrt{{[/\\]}}arm64
# CHECK-NEXT:   [[SYSROOT]]{{[/\\]}}sysroot{{[/\\]}}Windows Kits{{[/\\]}}10{{[/\\]}}Lib{{[/\\]}}10.0.1{{[/\\]}}um{{[/\\]}}arm64

        .data
        .word 1
