add_clang_library(MLIRCIRInterfacesCore
  OBJECT
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRTypeInterfaces.cpp

  PARTIAL_SOURCES_INTENDED

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
  MLIRPtrMemorySpaceInterfacesIncGen

  LINK_LIBS
  MLIRIRCore
  MLIRSupportCore

 )

add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp

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
  MLIRPtrMemorySpaceInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCIRInterfacesCore
  MLIRIRCore
  MLIRSupportCore

 )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRCIRInterfaces PRIVATE $<TARGET_OBJECTS:obj.MLIRCIRInterfacesCore>)
endif()
