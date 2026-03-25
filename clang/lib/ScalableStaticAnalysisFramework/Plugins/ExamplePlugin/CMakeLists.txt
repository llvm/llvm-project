# Do NOT set LLVM_LINK_COMPONENTS here. Setting it would cause LLVM libraries
# (e.g. Support) to be linked statically into this MODULE, producing a second
# copy of LLVM in the process alongside the copy already loaded by the host
# tool. Two copies means two separate llvm::Registry instances, so
# registrations made here would not be visible to the host tool.
#
# The same applies to clang static libraries: do not use
# clang_target_link_libraries here. Instead, expose only their include paths
# via $<TARGET_PROPERTY:...,INTERFACE_INCLUDE_DIRECTORIES> and let the
# dynamic loader resolve every SSAF and LLVM symbol from the tool's address
# space at load time.

add_llvm_library(SSAFExamplePlugin MODULE BUILDTREE_ONLY
  ExamplePlugin.cpp
  PLUGIN_TOOL clang-ssaf-format
  )

# Pull in SSAF and LLVM include paths without any static link dependency.
target_include_directories(SSAFExamplePlugin PRIVATE
  $<TARGET_PROPERTY:clangScalableStaticAnalysisFrameworkCore,INTERFACE_INCLUDE_DIRECTORIES>
  )
