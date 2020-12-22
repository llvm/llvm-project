set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  support
  )

add_clang_library(clangTidyAlteraModule
  AlteraTidyModule.cpp
  KernelNameRestrictionCheck.cpp
  SingleWorkItemBarrierCheck.cpp
  StructPackAlignCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyAlteraModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
