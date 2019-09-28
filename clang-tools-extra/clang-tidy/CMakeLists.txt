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
  GlobList.cpp

  DEPENDS
  ClangSACheckers

  LINK_LIBS
  clangAnalysis
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

# Checks.
# If you add a check, also add it to ClangTidyForceLinker.h in this directory.
add_subdirectory(android)
add_subdirectory(abseil)
add_subdirectory(boost)
add_subdirectory(bugprone)
add_subdirectory(cert)
add_subdirectory(cppcoreguidelines)
add_subdirectory(darwin)
add_subdirectory(fuchsia)
add_subdirectory(google)
add_subdirectory(hicpp)
add_subdirectory(linuxkernel)
add_subdirectory(llvm)
add_subdirectory(misc)
add_subdirectory(modernize)
if(CLANG_ENABLE_STATIC_ANALYZER)
  add_subdirectory(mpi)
endif()
add_subdirectory(objc)
add_subdirectory(openmp)
add_subdirectory(performance)
add_subdirectory(portability)
add_subdirectory(readability)
add_subdirectory(zircon)
set(ALL_CLANG_TIDY_CHECKS
  clangTidyAndroidModule
  clangTidyAbseilModule
  clangTidyBoostModule
  clangTidyBugproneModule
  clangTidyCERTModule
  clangTidyCppCoreGuidelinesModule
  clangTidyDarwinModule
  clangTidyFuchsiaModule
  clangTidyGoogleModule
  clangTidyHICPPModule
  clangTidyLinuxKernelModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyObjCModule
  clangTidyOpenMPModule
  clangTidyPerformanceModule
  clangTidyPortabilityModule
  clangTidyReadabilityModule
  clangTidyZirconModule
  )
if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND ALL_CLANG_TIDY_CHECKS clangTidyMPIModule)
endif()
set(ALL_CLANG_TIDY_CHECKS ${ALL_CLANG_TIDY_CHECKS} PARENT_SCOPE)

# Other subtargets. These may reference ALL_CLANG_TIDY_CHECKS
# and must be below its definition.
add_subdirectory(plugin)
add_subdirectory(tool)
add_subdirectory(utils)
