set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp

  LINK_LIBS
  clangCIR
  ${dialect_libs}
  MLIRCIR
  MLIRCIRTransforms
  MLIRTransforms
  MLIRSupport
  )

add_subdirectory(DirectToLLVM)
