set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp
  LoweringHelpers.cpp

  LINK_LIBS
  clangCIR
  ${dialect_libs}
  AIIRCIR
  AIIRCIRTransforms
  AIIRTransforms
  AIIRSupport
  )

add_subdirectory(DirectToLLVM)
