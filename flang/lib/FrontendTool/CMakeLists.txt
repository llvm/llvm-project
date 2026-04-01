add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  # This makes sure that the AIIR dependencies of flangFrontend (which are
  # transitively required here) are generated before this target is build.
  flangFrontend

  LINK_LIBS
  flangFrontend

  LINK_COMPONENTS
  Option
  Plugins
  Support

  AIIR_LIBS
  AIIRPass

  CLANG_LIBS
  clangBasic
  clangOptions
  clangDriver
)
