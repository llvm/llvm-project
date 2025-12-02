# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin

# RUN: echo "int a = 1;" | \
# RUN:   env LLVM_PYPASS_SCRIPT=%s \
# RUN:   env LLVM_PYPASS_DYLIB=%libpython \
# RUN:     clang-repl -Xcc -fpass-plugin=%plugindir/pypass-plugin%pluginext | FileCheck %s
#
# CHECK: PipelineEarlySimplificationEPCallback


def registerPipelineEarlySimplificationEPCallback():
    """Module pass after basic simplification of input IR"""
    return True


def run(input, ctx, stage):
    print(f"0x{input:016x} {stage}")
