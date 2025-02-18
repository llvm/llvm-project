add_llvm_library(MicrosoftAttributes MODULE MicrosoftAttributes.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  target_link_libraries(MicrosoftAttributes PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    LLVMSupport
    )
endif()
