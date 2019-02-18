set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangIncludeFixer
  IncludeFixer.cpp
  IncludeFixerContext.cpp
  InMemorySymbolIndex.cpp
  FuzzySymbolIndex.cpp
  SymbolIndexManager.cpp
  YamlSymbolIndex.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangParse
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  findAllSymbols
  )

add_subdirectory(plugin)
add_subdirectory(tool)
add_subdirectory(find-all-symbols)
