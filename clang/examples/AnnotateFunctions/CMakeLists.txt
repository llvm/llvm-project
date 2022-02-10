add_llvm_library(AnnotateFunctions MODULE AnnotateFunctions.cpp PLUGIN_TOOL clang)

if(CLANG_PLUGIN_SUPPORT AND (WIN32 OR CYGWIN))
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(AnnotateFunctions PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    )
endif()
