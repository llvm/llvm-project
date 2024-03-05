set(LLVM_LINK_COMPONENTS
  Support
  TextAPI
  Core
  )

add_clang_library(clangInstallAPI
  FileList.cpp
  Frontend.cpp
  HeaderFile.cpp
  Visitor.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )
