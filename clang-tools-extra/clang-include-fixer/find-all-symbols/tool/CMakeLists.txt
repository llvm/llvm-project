include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(find-all-symbols
  FindAllSymbolsMain.cpp
  )

clang_target_link_libraries(find-all-symbols
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )
target_link_libraries(find-all-symbols
  PRIVATE
  findAllSymbols
  )

install(PROGRAMS run-find-all-symbols.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT find-all-symbols)
