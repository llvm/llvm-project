add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  # This makes sure that the MLIR dependencies of flangFrontend (which are
  # transitively required here) are generated before this target is build.
  flangFrontend

  LINK_LIBS
  flangFrontend
  MLIRPass

  LINK_COMPONENTS
  Option
  Support

  CLANG_LIBS
  clangBasic
  clangDriver
)
