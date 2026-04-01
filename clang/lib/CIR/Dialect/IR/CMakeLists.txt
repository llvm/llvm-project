add_clang_library(AIIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp

  DEPENDS
  AIIRCIROpsIncGen
  AIIRCIRTypeConstraintsIncGen
  AIIRCIREnumsGen
  AIIRCIROpInterfacesIncGen
  AIIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRCIRInterfaces
  AIIRDLTIDialect
  AIIRDataLayoutInterfaces
  AIIRFuncDialect
  AIIRLoopLikeInterface
  AIIRCIRInterfaces
  clangAST
  )
