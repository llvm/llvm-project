set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(INCLUDE_FIXER_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include-fixer/find-all-symbols REALPATH)
include_directories(
  ${INCLUDE_FIXER_SOURCE_DIR}
  )

add_extra_unittest(FindAllSymbolsTests
  FindAllSymbolsTests.cpp
  )

target_link_libraries(FindAllSymbolsTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  findAllSymbols
  )
