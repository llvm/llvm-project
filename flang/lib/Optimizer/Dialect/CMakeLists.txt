get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp

  DEPENDS
  FIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRSupport
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIREDSC
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRStandardToLLVM
  MLIRTransforms

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
