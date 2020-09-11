add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendOptions.cpp

  LINK_LIBS
  clangBasic
  clangDriver
  # TODO: Added to re-use clang's TextDiagnosticBuffer & TextDiagnosticPrinter.
  # Add a custom implementation for Flang and remove this dependency.
  clangFrontend

  LINK_COMPONENTS
  Option
  Support
)
