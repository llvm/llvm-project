add_clang_library(clangTidyPlugin
  ClangTidyPlugin.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSema
  clangTidy
  clangTidyAbseilModule
  clangTidyAndroidModule
  clangTidyBoostModule
  clangTidyBugproneModule
  clangTidyCERTModule
  clangTidyCppCoreGuidelinesModule
  clangTidyFuchsiaModule
  clangTidyGoogleModule
  clangTidyHICPPModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyObjCModule
  clangTidyOpenMPModule
  clangTidyPerformanceModule
  clangTidyPortabilityModule
  clangTidyReadabilityModule
  clangTidyZirconModule
  clangTooling
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clangTidyPlugin PRIVATE
    clangTidyMPIModule
  )
endif()
