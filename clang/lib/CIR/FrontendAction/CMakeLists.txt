set(LLVM_LINK_COMPONENTS
  Core
  ipo
  Linker
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# CIRGenAction reaches into the Transforms-private LowerModule helper to
# stand up the AST-free LoweringPrepare pipeline.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Transforms)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp
  CIRDiagnosticHandler.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIROpInterfacesIncGen
  MLIRBuiltinOpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangCIR
  clangCIRLoweringCommon
  clangCIRLoweringDirectToLLVM
  clangCodeGen
  CIROpenACCSupport
  CIROpenMPSupport
  MLIRCIR
  MLIRCIRTargetLowering
  MLIRCIRTransforms
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRParser
  )
