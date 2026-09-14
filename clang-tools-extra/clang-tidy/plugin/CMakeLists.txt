if(WIN32 OR NOT CLANG_PLUGIN_SUPPORT OR NOT CLANG_LINK_CLANG_DYLIB OR
   NOT LLVM_LINK_LLVM_DYLIB)
  return()
endif()

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ClangTidyPlugin.exports)
add_llvm_library(clangTidyPlugin MODULE
  ClangTidyPlugin.cpp

  PLUGIN_TOOL clang

  LINK_LIBS
  clangTidy
  ${ALL_CLANG_TIDY_CHECKS}

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyPlugin
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSema
  clangTooling
  )
