add_llvm_loadable_module(AnnotateFunctions AnnotateFunctions.cpp)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  target_link_libraries(AnnotateFunctions ${cmake_2_8_12_PRIVATE}
    clangAST
    clangFrontend
    LLVMSupport
    )
endif()
