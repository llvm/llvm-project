set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  support
  )

add_clang_tool(clang-tidy
  ClangTidyMain.cpp
  )
add_dependencies(clang-tidy
  clang-resource-headers
  )
target_link_libraries(clang-tidy
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  clangTidyAndroidModule
  clangTidyAbseilModule
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
  clangToolingCore
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clang-tidy PRIVATE
    clangTidyMPIModule
  )
endif()

install(PROGRAMS clang-tidy-diff.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
install(PROGRAMS run-clang-tidy.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
