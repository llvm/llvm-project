set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
  ClangTidyDiagnosticConsumer.cpp
  ClangTidyOptions.cpp

  DEPENDS
  ClangSACheckers

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangRewrite
  clangSema
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  clangTooling
  clangToolingCore
  )

add_subdirectory(android)
add_subdirectory(boost)
add_subdirectory(bugprone)
add_subdirectory(cert)
add_subdirectory(cppcoreguidelines)
add_subdirectory(fuchsia)
add_subdirectory(google)
add_subdirectory(hicpp)
add_subdirectory(llvm)
add_subdirectory(misc)
add_subdirectory(modernize)
add_subdirectory(mpi)
add_subdirectory(objc)
add_subdirectory(performance)
add_subdirectory(plugin)
add_subdirectory(readability)
add_subdirectory(tool)
add_subdirectory(utils)
