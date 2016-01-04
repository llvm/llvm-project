set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_executable(clang-tidy
  ClangTidyMain.cpp
  )
target_link_libraries(clang-tidy
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  clangTidyCERTModule
  clangTidyCppCoreGuidelinesModule
  clangTidyGoogleModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyPerformanceModule
  clangTidyReadabilityModule
  clangTooling
  )

install(TARGETS clang-tidy
  RUNTIME DESTINATION bin)

install(PROGRAMS clang-tidy-diff.py DESTINATION share/clang)
install(PROGRAMS run-clang-tidy.py DESTINATION share/clang)
