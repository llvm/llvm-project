set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangIncludeFixer STATIC
  IncludeFixer.cpp
  IncludeFixerContext.cpp
  InMemorySymbolIndex.cpp
  FuzzySymbolIndex.cpp
  SymbolIndexManager.cpp
  YamlSymbolIndex.cpp

  LINK_LIBS
  findAllSymbols

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangIncludeFixer
  PRIVATE
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
  )

add_subdirectory(plugin)
add_subdirectory(tool)
add_subdirectory(find-all-symbols)
