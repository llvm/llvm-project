set(LLVM_LINK_COMPONENTS
  Support
  FrontendOpenMP
  )

add_clang_library(findAllSymbols
  FindAllSymbols.cpp
  FindAllSymbolsAction.cpp
  FindAllMacros.cpp
  HeaderMapCollector.cpp
  PathConfig.cpp
  PragmaCommentHandler.cpp
  STLPostfixHeaderMap.cpp
  SymbolInfo.cpp
  )

clang_target_link_libraries(findAllSymbols
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  )

add_subdirectory(tool)
