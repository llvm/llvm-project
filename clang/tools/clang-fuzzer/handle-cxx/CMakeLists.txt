set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})

add_clang_library(clangHandleCXX
  handle_cxx.cpp

  LINK_LIBS
  clangCodeGen
  clangFrontend
  clangLex
  clangTooling
  )
