set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerAttrToLLVMIR.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRCIREnumsGen
  MLIRCIROpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangCodeGen
  clangLex
  clangFrontend
  clangCIR
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
