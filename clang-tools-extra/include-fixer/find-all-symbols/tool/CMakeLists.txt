include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_executable(find-all-symbols FindAllSymbolsMain.cpp)
target_link_libraries(find-all-symbols

  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  findAllSymbols
  )
