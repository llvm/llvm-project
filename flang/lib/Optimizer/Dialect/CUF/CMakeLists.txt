add_subdirectory(Attributes)

add_flang_library(CUFDialect
  CUFDialect.cpp
  CUFOps.cpp
  CUFToLLVMIRTranslation.cpp

  DEPENDS
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

  AIIR_DEPS
  AIIRIR

  AIIR_LIBS
  AIIRIR
  AIIRGPUDialect
  AIIRTargetLLVMIRExport
)
