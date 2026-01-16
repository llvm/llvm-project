# There's no reason to export anything from the plugin.
set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/PrintFunctionNames.exports)

add_llvm_library(PrintFunctionNames MODULE PrintFunctionNames.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(PrintFunctionNames PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()
