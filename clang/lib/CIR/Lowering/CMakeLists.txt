set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# runCIRToCIRPasses reaches into the Transforms-private LowerModule helper to
# read the target/LangOpts state that drives CallConvLowering, so the pipeline
# needs no live ASTContext.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Transforms)

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp
  LoweringHelpers.cpp

  LINK_LIBS
  clangCIR
  ${dialect_libs}
  MLIRCIR
  MLIRCIRTransforms
  MLIRTransforms
  MLIRSupport
  )

add_subdirectory(DirectToLLVM)
