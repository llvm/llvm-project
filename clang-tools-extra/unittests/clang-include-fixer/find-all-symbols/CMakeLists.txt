set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

get_filename_component(INCLUDE_FIXER_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../clang-include-fixer/find-all-symbols REALPATH)
include_directories(
  ${INCLUDE_FIXER_SOURCE_DIR}
  )

add_extra_unittest(FindAllSymbolsTests
  FindAllSymbolsTests.cpp
  )

clang_target_link_libraries(FindAllSymbolsTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )
target_link_libraries(FindAllSymbolsTests
  PRIVATE
  findAllSymbols
  )
