set(LLVM_LINK_COMPONENTS
  Support
  TextAPI
  Demangle
  Core
  )

add_clang_library(clangInstallAPI
  DylibVerifier.cpp
  FileList.cpp
  Frontend.cpp
  HeaderFile.cpp
  Visitor.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )
