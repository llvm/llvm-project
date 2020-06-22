set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyBoostModule
  BoostTidyModule.cpp
  UseToStringCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyBoostModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
