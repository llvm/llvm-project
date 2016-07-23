set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANG_RENAME_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-rename REALPATH)
include_directories(
  ${CLANG_RENAME_SOURCE_DIR}
  )

add_extra_unittest(ClangRenameTests
  USRLocFindingTest.cpp
  ${CLANG_RENAME_SOURCE_DIR}/USRFinder.cpp
  ${CLANG_RENAME_SOURCE_DIR}/USRFindingAction.cpp
  )

target_link_libraries(ClangRenameTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangTooling
  )
