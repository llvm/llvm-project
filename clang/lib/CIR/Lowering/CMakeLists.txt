set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp
  LoweringHelpers.cpp

  DEPENDS
  MLIRCIRPassIncGen
  mlir-headers

  LINK_LIBS
  clangCIR
  MLIRCIR
  MLIRCIRTransforms
  MLIRTransforms
  MLIRSupport
  )

add_subdirectory(DirectToLLVM)
