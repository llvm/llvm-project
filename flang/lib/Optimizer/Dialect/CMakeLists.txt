add_subdirectory(Support)
add_subdirectory(CUF)
add_subdirectory(FIRCG)
add_subdirectory(MIF)

add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  FirAliasTagOpInterface.cpp
  FortranVariableInterface.cpp
  Inliner.cpp
  SafeTempArrayCopyAttrInterface.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  FIROpsIncGen
  FIRSafeTempArrayCopyAttrInterfaceIncGen
  CUFAttrsIncGen
  intrinsics_gen

  LINK_LIBS
  CUFAttrs
  FIRDialectSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
)
