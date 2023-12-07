set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyAndroidModule
  AndroidTidyModule.cpp
  CloexecAccept4Check.cpp
  CloexecAcceptCheck.cpp
  CloexecCheck.cpp
  CloexecCreatCheck.cpp
  CloexecDupCheck.cpp
  CloexecEpollCreate1Check.cpp
  CloexecEpollCreateCheck.cpp
  CloexecFopenCheck.cpp
  CloexecInotifyInit1Check.cpp
  CloexecInotifyInitCheck.cpp
  CloexecMemfdCreateCheck.cpp
  CloexecOpenCheck.cpp
  CloexecPipeCheck.cpp
  CloexecPipe2Check.cpp
  CloexecSocketCheck.cpp
  ComparisonInTempFailureRetryCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyAndroidModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
