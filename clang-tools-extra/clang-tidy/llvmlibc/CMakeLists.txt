set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMLibcModule
  CalleeNamespaceCheck.cpp
  ImplementationInNamespaceCheck.cpp
  LLVMLibcTidyModule.cpp
  RestrictSystemLibcHeadersCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyPortabilityModule
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyLLVMLibcModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
