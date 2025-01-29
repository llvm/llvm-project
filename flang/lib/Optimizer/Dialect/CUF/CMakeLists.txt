add_subdirectory(Attributes)

add_flang_library(CUFDialect
  CUFDialect.cpp
  CUFOps.cpp
  CUFToLLVMIRTranslation.cpp

  DEPENDS
  MLIRIR
  CUFAttrsIncGen
  CUFOpsIncGen

  LINK_LIBS
  CUFAttrs
  FIRDialect
  FIRDialectSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_LIBS
  MLIRIR
  MLIRGPUDialect
  MLIRTargetLLVMIRExport
)
