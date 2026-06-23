# CMakeLists.txt for Instrumentor Examples
#
# This directory contains example runtimes that demonstrate how to use the
# LLVM Instrumentor pass for various profiling and analysis tasks.

include(AddCompilerRT)

add_compiler_rt_component(instrumentor-tools)

# Add subdirectories for specific examples
add_subdirectory(flop-counter)
add_subdirectory(fp-precision-analysis)
add_subdirectory(pointer-tracking)
