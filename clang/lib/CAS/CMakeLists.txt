set(LLVM_LINK_COMPONENTS
  Support
  CAS
  )

add_clang_library(clangCAS
  CASOptions.cpp

  LINK_LIBS
  clangBasic
  )

