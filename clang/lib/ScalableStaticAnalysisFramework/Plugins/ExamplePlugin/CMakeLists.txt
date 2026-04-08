# Do NOT set LLVM_LINK_COMPONENTS and clang_target_link_libraries here.
#
# Setting them would cause LLVM libraries to be linked statically into this
# MODULE, producing a second copy of LLVM in the process alongside the copy
# already loaded by the host tool. Let the dynamic loader resolve every SSAF and
# LLVM symbol from the tool's address space at load time.

add_llvm_library(SSAFExamplePlugin MODULE BUILDTREE_ONLY
  TagsAnalysis.cpp
  PairsAnalysis.cpp
  TagsPairsAnalysis.cpp
  PLUGIN_TOOL clang-ssaf-format
  )

# Pull in SSAF and LLVM include paths without any static link dependency.
target_include_directories(SSAFExamplePlugin PRIVATE
  $<TARGET_PROPERTY:clangScalableStaticAnalysisFrameworkCore,INTERFACE_INCLUDE_DIRECTORIES>
  )
