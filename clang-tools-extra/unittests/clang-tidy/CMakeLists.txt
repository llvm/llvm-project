set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANG_LINT_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-tidy REALPATH)
include_directories(${CLANG_LINT_SOURCE_DIR})

add_extra_unittest(ClangTidyTests
  ClangTidyDiagnosticConsumerTest.cpp
  ClangTidyOptionsTest.cpp
  IncludeInserterTest.cpp
  GlobListTest.cpp
  GoogleModuleTest.cpp
  LLVMModuleTest.cpp
  NamespaceAliaserTest.cpp
  ObjCModuleTest.cpp
  OverlappingReplacementsTest.cpp
  UsingInserterTest.cpp
  ReadabilityModuleTest.cpp
  TransformerClangTidyCheckTest.cpp
  )

clang_target_link_libraries(ClangTidyTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  clangTransformer
  )
target_link_libraries(ClangTidyTests
  PRIVATE
  clangTidy
  clangTidyAndroidModule
  clangTidyGoogleModule
  clangTidyLLVMModule
  clangTidyObjCModule
  clangTidyReadabilityModule
  clangTidyUtils
  )
