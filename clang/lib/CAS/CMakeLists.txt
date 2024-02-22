set(LLVM_LINK_COMPONENTS
  Support
  CAS
  )

add_clang_library(clangCAS
  CASOptions.cpp
  IncludeTree.cpp

  LINK_LIBS
  clangBasic
  )

