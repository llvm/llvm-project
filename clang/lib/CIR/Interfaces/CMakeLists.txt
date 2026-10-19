add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIREnumsGen
  MLIRCIRTypeInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIROpsIncGen
  MLIRCIRTypeConstraintsIncGen
  # CIR interface headers include generated core MLIR interfaces without
  # linking all of their owning libraries.
  MLIRCallInterfacesIncGen
  MLIRControlFlowInterfacesIncGen
  MLIRFunctionInterfacesIncGen
  MLIRInferTypeOpInterfaceIncGen
  MLIRLoopLikeInterfaceIncGen
  MLIRSideEffectInterfacesIncGen

  LINK_LIBS
  MLIRIR
  MLIRPtrMemorySpaceInterfaces
  MLIRSupport
 )
