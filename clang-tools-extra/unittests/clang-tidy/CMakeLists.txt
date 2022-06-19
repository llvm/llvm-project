set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

if(CLANG_BUILT_STANDALONE)
  # LLVMTestingSupport library is needed for clang-tidy tests.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

get_filename_component(CLANG_LINT_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-tidy REALPATH)
include_directories(${CLANG_LINT_SOURCE_DIR})

add_extra_unittest(ClangTidyTests
  AddConstTest.cpp
  ClangTidyDiagnosticConsumerTest.cpp
  ClangTidyOptionsTest.cpp
  DeclRefExprUtilsTest.cpp
  IncludeInserterTest.cpp
  GlobListTest.cpp
  GoogleModuleTest.cpp
  LLVMModuleTest.cpp
  ModernizeModuleTest.cpp
  NamespaceAliaserTest.cpp
  ObjCModuleTest.cpp
  OptionsProviderTest.cpp
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
  clangTesting
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
  clangTidyModernizeModule
  clangTidyObjCModule
  clangTidyReadabilityModule
  clangTidyUtils
  LLVMTestingSupport
  )
