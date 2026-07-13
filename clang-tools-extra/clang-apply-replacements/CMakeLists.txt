set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangApplyReplacements STATIC
  lib/Tooling/ApplyReplacements.cpp

  DEPENDS
  ClangDriverOptions
)

clang_target_link_libraries(clangApplyReplacements
  PRIVATE
  clangAST
  clangBasic
  clangRewrite
  clangToolingCore
  clangToolingRefactoring
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  include
  )
add_subdirectory(tool)
