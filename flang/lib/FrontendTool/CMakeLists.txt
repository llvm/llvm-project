add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  # This makes sure that the MLIR dependencies of flangFrontend (which are
  # transitively required here) are generated before this target is build.
  flangFrontend
  clangBasic

  LINK_LIBS
  flangFrontend
  clangBasic
  clangDriver

  LINK_COMPONENTS
  Option
  Support
)
