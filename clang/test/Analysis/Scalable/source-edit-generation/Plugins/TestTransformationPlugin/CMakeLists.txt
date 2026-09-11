# Do NOT set LLVM_LINK_COMPONENTS or clang_target_link_libraries here.
#
# Static link would produce a second copy of LLVM and Clang libraries in the
# plugin's address space alongside the copies already loaded by clang itself,
# resulting in two distinct llvm::Registry instances and broken registration.
# Let the dynamic loader resolve every symbol from clang at load time.

add_llvm_library(SSAFTestTransformationPlugin MODULE BUILDTREE_ONLY
  TestTransformation.cpp
  PLUGIN_TOOL clang
  )

target_include_directories(SSAFTestTransformationPlugin PRIVATE
  $<TARGET_PROPERTY:clangScalableStaticAnalysisCore,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:clangScalableStaticAnalysisSourceTransformation,INTERFACE_INCLUDE_DIRECTORIES>
  )
