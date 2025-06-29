add_llvm_library(JumboSupport MODULE JumboSupport.cpp PLUGIN_TOOL clang)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(JumboSupport PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    LLVMSupport
    )
endif()
