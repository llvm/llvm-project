# REQUIRES: x86
## Test we ignore some LTO related options from clang/GCC collect2.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## GCC collect2 passes several LTO related options to the linker even if -flto is not used.
## We need to ignore them. Note that the lto-wrapper path can be relative.
# RUN: ld.lld %t.o -o /dev/null \
# RUN:   -plugin path/to/liblto_plugin.so \
# RUN:   -plugin-opt=/path/to/lto-wrapper \
# RUN:   -plugin-opt=/path/to/lto-wrapper.exe \
# RUN:   -plugin-opt=relative/path/to/lto-wrapper \
# RUN:   -plugin-opt=relative/path/to/lto-wrapper.exe \
# RUN:   -plugin-opt=-fresolution=zed \
# RUN:   -plugin-opt=-pass-through=-lgcc \
# RUN:   -plugin-opt=-pass-through=-lgcc_eh \
# RUN:   -plugin-opt=-pass-through=-lc

## Clang LTO passes several options to the linker, which are intended to be consumed by
## LLVMgold.so. We need to ignore them.
# RUN: ld.lld %t.o -o /dev/null -plugin /path/to/LLVMgold.so -plugin-opt=thinlto

## Other -plugin-opt=- prefixed options are passed through to cl::ParseCommandLineOptions.
# RUN: not ld.lld %t.o -o /dev/null -plugin-opt=-abc -plugin-opt=-xyz 2>&1 | FileCheck %s
# CHECK: ld.lld: error: -plugin-opt=-: ld.lld{{.*}}: Unknown command line argument '-abc'
# CHECK: ld.lld: error: -plugin-opt=-: ld.lld{{.*}}: Unknown command line argument '-xyz'

## Error if the option is an unhandled LLVMgold.so feature.
# RUN: not ld.lld %t.o -o /dev/null -plugin-opt=LLVMgold-feature 2>&1 | FileCheck --check-prefix=GOLD %s
# GOLD: ld.lld: error: -plugin-opt=: unknown plugin option 'LLVMgold-feature'
