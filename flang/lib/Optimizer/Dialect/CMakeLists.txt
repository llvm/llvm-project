add_subdirectory(Support)

add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  FortranVariableInterface.cpp
  Inliner.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  MLIRIR
  FIROpsIncGen

  LINK_LIBS
  FIRDialectSupport
  MLIRArithDialect
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
