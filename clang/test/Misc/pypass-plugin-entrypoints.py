# REQUIRES: native, system-linux, plugins, pypass-plugin

# Entry-points in default and -O0 pipeline
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP %s
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -flto=full -O0 \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP %s
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -flto=thin -O0 \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP %s
#
# EP-NOT: PeepholeEPCallback
# EP-NOT: Optimizer{{.*}}EPCallback
# EP-NOT: ScalarOptimizer{{.*}}EPCallback
# EP-NOT: FullLinkTimeOptimization{{.*}}EPCallback
#
# EP: PipelineStartEPCallback
# EP: PipelineEarlySimplificationEPCallback
# EP: OptimizerEarlyEPCallback
# EP: OptimizerLastEPCallback

# Entry-points in optimizer pipeline
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -O2 \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP-OPT %s
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -O2 -flto=full \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP-OPT %s
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -O2 -ffat-lto-objects \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP-OPT %s
#
# EP-OPT:     PipelineStartEPCallback
# EP-OPT:     PipelineEarlySimplificationEPCallback
# EP-OPT:     PeepholeEPCallback
# EP-OPT:     ScalarOptimizerLateEPCallback
# EP-OPT:     PeepholeEPCallback
# EP-OPT:     OptimizerEarlyEPCallback
# EP-OPT:     VectorizerStartEPCallback
# EP-OPT:     VectorizerEndEPCallback
# EP-OPT:     OptimizerLastEPCallback

# FIXME: Thin-LTO does not invoke vectorizer callbacks
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   %clang -fpass-plugin=%plugindir/pypass-plugin%pluginext -O2 -flto=thin \
# RUN:          -o /dev/null -S -emit-llvm %S/Inputs/pypass-plugin.c | FileCheck --check-prefix=EP-LTO-THIN %s
#
# EP-LTO-THIN:     PipelineStartEPCallback
# EP-LTO-THIN:     PipelineEarlySimplificationEPCallback
# EP-LTO-THIN:     PeepholeEPCallback
# EP-LTO-THIN:     ScalarOptimizerLateEPCallback
# EP-LTO-THIN:     OptimizerEarlyEPCallback
# EP-LTO-THIN-NOT: Vectorizer{{.*}}EPCallback
# EP-LTO-THIN:     OptimizerLastEPCallback


def registerPipelineStartEPCallback():
    """Module pass at the start of the pipeline"""
    return True


def registerPipelineEarlySimplificationEPCallback():
    """Module pass after basic simplification of input IR"""
    return True


def registerOptimizerEarlyEPCallback():
    """Module pass before the function optimization pipeline"""
    return True


def registerOptimizerLastEPCallback():
    """Module pass after the function optimization pipeline"""
    return True


def registerPeepholeEPCallback():
    """Function pass after each instance of the instruction combiner pass"""
    return True


def registerScalarOptimizerLateEPCallback():
    """Function pass after most of the main optimizations, but before the last
    cleanup-ish optimizations"""
    return True


def registerVectorizerStartEPCallback():
    """Function pass before the vectorizer and other highly target specific
    optimization passes are executed"""
    return True


def registerVectorizerEndEPCallback():
    """Function pass after the vectorizer and other highly target specific
    optimization passes are executed"""
    return True


def registerFullLinkTimeOptimizationEarlyEPCallback():
    """Module pass at the start of the full LTO pipeline"""
    return True


def registerFullLinkTimeOptimizationLastEPCallback():
    """Module pass at the end of the full LTO pipeline"""
    return True


def run(input, ctx, stage):
    print(f"0x{input:016x} {stage}")
