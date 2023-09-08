set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  support
  )

add_clang_library(clangTidyAlteraModule
  AlteraTidyModule.cpp
  IdDependentBackwardBranchCheck.cpp
  KernelNameRestrictionCheck.cpp
  SingleWorkItemBarrierCheck.cpp
  StructPackAlignCheck.cpp
  UnrollLoopsCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyAlteraModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
