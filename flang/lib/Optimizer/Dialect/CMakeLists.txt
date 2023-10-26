add_subdirectory(Support)

add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  FortranVariableInterface.cpp
  FirAliasTagOpInterface.cpp
  Inliner.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  MLIRIR
  FIROpsIncGen
  intrinsics_gen

  LINK_LIBS
  FIRDialectSupport
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
