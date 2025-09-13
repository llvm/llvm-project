set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangCodeGenShared
  ItaniumRTTIBuilder.cpp

  DEPENDS
  LINK_LIBS
  clangAST
  )
