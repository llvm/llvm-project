set(LLVM_LINK_COMPONENTS
  Support
  TextAPI
  )

add_clang_library(clangInstallAPI
  FileList.cpp
  HeaderFile.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
