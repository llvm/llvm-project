# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin
#
# RUN: opt %S/Inputs/pypass-plugin.ll -o %t.o
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   ld.lld --load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:          --lto-newpm-passes=pypass %t.o -o /dev/null | FileCheck %s
#
# CHECK: 0x{{[0-9a-f]+}} Module


def registerModulePipelineParsingCallback():
    return True


def run(input, ctx, stage):
    print(f"0x{input:016x} {stage}")
