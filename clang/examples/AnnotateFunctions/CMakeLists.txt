add_llvm_loadable_module(AnnotateFunctions AnnotateFunctions.cpp PLUGIN_TOOL clang)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  target_link_libraries(AnnotateFunctions PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    LLVMSupport
    )
endif()
