set(LLVM_LINK_COMPONENTS
  Option
  Support
)

add_clang_library(clangOptions
  DriverOptions.cpp
  OptionUtils.cpp
  
  DEPENDS
  ClangDriverOptions
  # These generated headers are included transitively.
  target_parser_gen

  LINK_LIBS
  clangBasic
  ${system_libs}
)
