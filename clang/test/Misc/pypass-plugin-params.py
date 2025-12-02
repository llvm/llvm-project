# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin

# XFAIL: *
#
# RUN: %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -S -emit-llvm \
# RUN:        -mllvm -pypass-script=%s \
# RUN:        -mllvm -pypass-dylib=%libpython \
# RUN:        -Xclang -fdebug-pass-manager \
# RUN:        -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c 2>&1 | FileCheck %s
#
# CHECK: Unknown command line argument

# Plugin parameters only work with the extra `-Xclang -load -Xclang <path>`
# RUN: %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -S -emit-llvm \
# RUN:        -Xclang -load -Xclang %plugindir/pypass-plugin%pluginext \
# RUN:        -mllvm -pypass-script=%s \
# RUN:        -mllvm -pypass-dylib=%libpython \
# RUN:        -Xclang -fdebug-pass-manager \
# RUN:        -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c 2>&1 | FileCheck %s
#
# CHECK: Running pass: PyPass


def registerPipelineEarlySimplificationEPCallback():
    """Module pass after basic simplification of input IR"""
    return True


def run(input, ctx, stage):
    print(f"0x{input:016x} {stage}")
