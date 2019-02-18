set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangApplyReplacements
  lib/Tooling/ApplyReplacements.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangRewrite
  clangToolingCore
  clangToolingRefactor
  clangToolingRefactoring
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  include
  )
add_subdirectory(tool)
