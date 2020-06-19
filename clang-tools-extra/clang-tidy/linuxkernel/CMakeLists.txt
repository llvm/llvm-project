set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLinuxKernelModule
  LinuxKernelTidyModule.cpp
  MustCheckErrsCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyLinuxKernelModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
