# There's no reason to export anything from the plugin.
set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/LLVMPrintFunctionNames.exports)

add_llvm_library(LLVMPrintFunctionNames MODULE LLVMPrintFunctionNames.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(LLVMPrintFunctionNames PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()
