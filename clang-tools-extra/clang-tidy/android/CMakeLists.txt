set(LLVM_LINK_COMPONENTS support)

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
  CloexecSocketCheck.cpp
  ComparisonInTempFailureRetryCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
