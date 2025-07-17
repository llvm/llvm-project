set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMModule STATIC
  HeaderGuardCheck.cpp
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  MLIROpBuilderCheck.cpp
  PreferIsaOrDynCastInConditionalsCheck.cpp
  PreferRegisterOverUnsignedCheck.cpp
  PreferStaticOverAnonymousNamespaceCheck.cpp
  TwineLocalCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils
  clangTransformer

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
