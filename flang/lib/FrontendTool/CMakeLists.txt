add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  clangBasic

  LINK_LIBS
  clangBasic
  clangDriver

  LINK_COMPONENTS
  Option
  Support
)
