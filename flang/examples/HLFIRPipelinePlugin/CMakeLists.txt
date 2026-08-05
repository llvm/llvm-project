# TODO: Note that this is currently only available on Linux.
# On Windows, we would also have to specify e.g. `PLUGIN_TOOL`.
#
# Nothing is linked in on purpose: the MLIR, FIR and flang symbols the plugin
# uses are resolved against the host tool (`flang -fc1` or `fir-opt`) when the
# shared object is loaded, which is why both export their symbols for plugins.
add_llvm_example_library(flangHLFIRPipelinePlugin
    MODULE
    HLFIRPipelinePlugin.cpp

    DEPENDS
    acc_gen
    flangPasses
    omp_gen
)
