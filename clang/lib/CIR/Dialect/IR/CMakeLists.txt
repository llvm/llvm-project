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
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  clangAST
  )
