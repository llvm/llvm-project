add_llvm_library(Attribute MODULE Attribute.cpp PLUGIN_TOOL clang)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  target_link_libraries(AnnotateFunctions ${cmake_2_8_12_PRIVATE}
    clangAST
    clangBasic
    clangFrontend
    clangLex
    LLVMSupport
    )
endif()
