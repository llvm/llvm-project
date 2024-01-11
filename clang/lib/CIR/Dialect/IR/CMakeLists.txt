add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRTypes.cpp
  FPEnv.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
  MLIRSymbolInterfacesIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  clangAST
  )
