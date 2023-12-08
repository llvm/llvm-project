set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyObjCModule
  AssertEquals.cpp
  AvoidNSErrorInitCheck.cpp
  DeallocInCategoryCheck.cpp
  ForbiddenSubclassingCheck.cpp
  MissingHashCheck.cpp
  NSInvocationArgumentLifetimeCheck.cpp
  NSDateFormatterCheck.cpp
  ObjCTidyModule.cpp
  PropertyDeclarationCheck.cpp
  SuperSelfCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyObjCModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
