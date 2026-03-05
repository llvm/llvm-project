# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin

# Entry-points in pipeline for regular/monolithic LTO
#
# RUN: opt %S/Inputs/pypass-plugin.ll -o %t.o
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   ld.lld --load-pass-plugin=%plugindir/pypass-plugin%pluginext %t.o \
# RUN:           -shared -o /dev/null | FileCheck --check-prefix=REGULAR %s
#
# REGULAR-NOT: PipelineStartEPCallback
# REGULAR-NOT: PipelineEarlySimplificationEPCallback
# REGULAR-NOT: PeepholeEPCallback
# REGULAR-NOT: ScalarOptimizerLateEPCallback
# REGULAR-NOT: Vectorizer{{.*}}EPCallback
# REGULAR-NOT: Optimizer{{.*}}EPCallback
#
# REGULAR: FullLinkTimeOptimizationEarlyEPCallback
# REGULAR: FullLinkTimeOptimizationLastEPCallback

# Entry-points in Thin-LTO pipeline
#
# RUN: opt --thinlto-bc %S/Inputs/pypass-plugin.ll -o %t_thin1.o
# RUN: opt -module-summary %S/Inputs/pypass-plugin.ll -o %t_thin2.bc
#
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN:   ld.lld --load-pass-plugin=%plugindir/pypass-plugin%pluginext %t_thin2.bc \
# RUN:           -shared -o /dev/null | FileCheck --check-prefix=THIN %s
#
# THIN-NOT: FullLinkTimeOptimizationEarlyEPCallback
# THIN-NOT: FullLinkTimeOptimizationLastEPCallback
# THIN-NOT: PipelineStartEPCallback
#
# THIN: PipelineEarlySimplificationEPCallback
# THIN: PeepholeEPCallback
# THIN: ScalarOptimizerLateEPCallback
# THIN: PeepholeEPCallback
# THIN: OptimizerEarlyEPCallback
# THIN: VectorizerStartEPCallback
# THIN: VectorizerEndEPCallback
# THIN: OptimizerLastEPCallback


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
