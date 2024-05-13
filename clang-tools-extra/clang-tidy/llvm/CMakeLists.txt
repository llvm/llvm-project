set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMModule
  HeaderGuardCheck.cpp
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  PreferIsaOrDynCastInConditionalsCheck.cpp
  PreferRegisterOverUnsignedCheck.cpp
  TwineLocalCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyLLVMModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
