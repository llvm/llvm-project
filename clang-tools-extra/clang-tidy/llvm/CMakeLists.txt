set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMModule STATIC
  HeaderGuardCheck.cpp
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  PreferIsaOrDynCastInConditionalsCheck.cpp
  PreferRegisterOverUnsignedCheck.cpp
  PreferStaticOverAnonymousNamespaceCheck.cpp
  TwineLocalCheck.cpp
  UseNewMLIROpBuilderCheck.cpp
  UseRangesCheck.cpp

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
  clangTransformer
  )
