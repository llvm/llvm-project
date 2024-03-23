set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-named-modules-querier
  ClangNamedModulesQuerier.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  GENERATE_DRIVER
  )

set(CLANG_NAMED_MODULES_QUERIER
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )

clang_target_link_libraries(clang-named-modules-querier
  PRIVATE
  ${CLANG_NAMED_MODULES_QUERIER}
  )

add_llvm_library(ClangGetUsedDeclsFromModulesPlugin PARTIAL_SOURCES_INTENDED MODULE GetUsedDeclsFromModulesPlugin.cpp PLUGIN_TOOL clang)

add_llvm_library(ClangGetUsedFilesFromModulesPlugin PARTIAL_SOURCES_INTENDED MODULE GetUsedFilesFromModulesPlugin.cpp PLUGIN_TOOL clang)
