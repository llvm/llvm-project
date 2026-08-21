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

# The plugin is loaded into a clang that may be built with sanitizers (e.g. the
# stage-2 -DLLVM_USE_SANITIZER=Address bots), but does not need instrumenting
# itself. With ASan, -Wl,-dead_strip drops anonymous constants that ASan's
# metadata still references, leaving dangling flat-namespace binds that break
# dlopen ("symbol not found in flat namespace 'l___unnamed_NNNN'"). Opt out.
if(LLVM_USE_SANITIZER)
  target_compile_options(SSAFTestTransformationPlugin PRIVATE -fno-sanitize=all)
  target_link_options(SSAFTestTransformationPlugin PRIVATE -fno-sanitize=all)
endif()
  target_compile_options(SSAFTestTransformationPlugin PRIVATE -fno-sanitize=all)
  target_link_options(SSAFTestTransformationPlugin PRIVATE -fno-sanitize=all)
endif()
