set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  DEPENDS
  MLIRCIROpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangCodeGen
  clangLex
  clangFrontend
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCIRLoweringThroughMLIR
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  )
