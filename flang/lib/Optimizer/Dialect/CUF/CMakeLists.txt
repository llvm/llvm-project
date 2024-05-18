add_subdirectory(Attributes)

add_flang_library(CUFDialect
  CUFDialect.cpp
  CUFOps.cpp

  DEPENDS
  MLIRIR
  CUFOpsIncGen

  LINK_LIBS
  CUFAttrs
  FIRDialect
  FIRDialectSupport
  MLIRIR
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
