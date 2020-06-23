set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyObjCModule
  AvoidNSErrorInitCheck.cpp
  DeallocInCategoryCheck.cpp
  ForbiddenSubclassingCheck.cpp
  MissingHashCheck.cpp
  NSInvocationArgumentLifetimeCheck.cpp
  ObjCTidyModule.cpp
  PropertyDeclarationCheck.cpp
  SuperSelfCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangTidyObjCModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
