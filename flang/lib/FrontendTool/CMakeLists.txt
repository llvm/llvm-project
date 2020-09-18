add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  LINK_LIBS
  clangBasic
  clangDriver

  LINK_COMPONENTS
  Option
  Support
)
