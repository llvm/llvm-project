set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyCheck.cpp
  ClangTidyModule.cpp
  ClangTidyDiagnosticConsumer.cpp
  ClangTidyOptions.cpp
  ClangTidyProfiling.cpp
  ExpandModularHeadersPPCallbacks.cpp

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
  clangSerialization
  clangTooling
  clangToolingCore
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clangTidy PRIVATE
    clangStaticAnalyzerCore
    clangStaticAnalyzerFrontend
  )
endif()

add_subdirectory(android)
add_subdirectory(abseil)
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
if(CLANG_ENABLE_STATIC_ANALYZER)
  add_subdirectory(mpi)
endif()
add_subdirectory(objc)
add_subdirectory(openmp)
add_subdirectory(performance)
add_subdirectory(plugin)
add_subdirectory(portability)
add_subdirectory(readability)
add_subdirectory(tool)
add_subdirectory(utils)
add_subdirectory(zircon)
