set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyOpenMPModule
  ExceptionEscapeCheck.cpp
  OpenMPTidyModule.cpp
  UseDefaultNoneCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyOpenMPModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  )
