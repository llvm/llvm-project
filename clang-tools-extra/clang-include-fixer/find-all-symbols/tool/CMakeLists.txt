include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(find-all-symbols
  FindAllSymbolsMain.cpp
  )

target_link_libraries(find-all-symbols
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  findAllSymbols
  )

install(PROGRAMS run-find-all-symbols.py
  DESTINATION share/clang
  COMPONENT find-all-symbols)
