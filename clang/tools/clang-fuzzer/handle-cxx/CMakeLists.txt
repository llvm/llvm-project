set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} Support)

add_clang_library(clangHandleCXX
  handle_cxx.cpp

  LINK_LIBS
  clangBasic
  clangCodeGen
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )
