set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangEdit
  Commit.cpp
  EditedSource.cpp
  FillInMissingProtocolStubs.cpp
  FillInMissingSwitchEnumCases.cpp
  RewriteObjCFoundationAPI.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )
