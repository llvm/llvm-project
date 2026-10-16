set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp
  LoweringHelpers.cpp

  LINK_LIBS
  clangCIR
  MLIRCIR
  MLIRCIRTransforms
  MLIRPass
  MLIRSupportCore
  )

add_subdirectory(DirectToLLVM)
