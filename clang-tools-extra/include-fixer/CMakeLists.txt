set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangIncludeFixer
  IncludeFixer.cpp
  InMemoryXrefsDB.cpp
  YamlXrefsDB.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangParse
  clangSema
  clangTooling
  clangToolingCore
  findAllSymbols
  )

add_subdirectory(tool)
add_subdirectory(find-all-symbols)
