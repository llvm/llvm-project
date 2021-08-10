add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  clang-cpp

  LINK_LIBS
  flangFrontend
  clang-cpp

  LINK_COMPONENTS
  Option
  Support
)
