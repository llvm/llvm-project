# Keep CUF attributes as a separate library as FIR and HLFIR depend on it. 
add_flang_library(CUFAttrs
  CUFAttr.cpp

  DEPENDS
  MLIRIR
  CUFAttrsIncGen

  LINK_LIBS
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
