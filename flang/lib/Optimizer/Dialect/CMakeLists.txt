add_subdirectory(Support)
add_subdirectory(CUF)
add_subdirectory(FIRCG)
add_subdirectory(MIF)

add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROperationMoveOpInterface.cpp
  FIROps.cpp
  FIRType.cpp
  FirAliasTagOpInterface.cpp
  FortranVariableInterface.cpp
  Inliner.cpp
  SafeTempArrayCopyAttrInterface.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  FIROperationMoveOpInterfaceIncGen
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

  AIIR_DEPS
  AIIRIR

  AIIR_LIBS
  AIIRArithDialect
  AIIROpenACCDialect
  AIIRBuiltinToLLVMIRTranslation
  AIIROpenMPToLLVM
  AIIRLLVMToLLVMIRTranslation
  AIIRTargetLLVMIRExport
)
