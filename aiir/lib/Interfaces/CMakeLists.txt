set(LLVM_OPTIONAL_SOURCES
  AlignmentAttrInterface.cpp
  CallInterfaces.cpp
  CastInterfaces.cpp
  ControlFlowInterfaces.cpp
  DataLayoutInterfaces.cpp
  DerivedAttributeOpInterface.cpp
  DestinationStyleOpInterface.cpp
  FunctionImplementation.cpp
  FunctionInterfaces.cpp
  IndexingMapOpInterface.cpp
  InferIntRangeInterface.cpp
  InferStridedMetadataInterface.cpp
  InferTypeOpInterface.cpp
  LoopLikeInterface.cpp
  MemOpInterfaces.cpp
  MemorySlotInterfaces.cpp
  ParallelCombiningOpInterface.cpp
  RuntimeVerifiableOpInterface.cpp
  ShapedOpInterfaces.cpp
  SideEffectInterfaces.cpp
  SubsetOpInterface.cpp
  TilingInterface.cpp
  ValueBoundsOpInterface.cpp
  VectorInterfaces.cpp
  ViewLikeInterface.cpp
  )

function(add_aiir_interface_library name)
  add_aiir_library(AIIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

    DEPENDS
    AIIR${name}IncGen

    LINK_LIBS PUBLIC
    AIIRIR
    )
endfunction(add_aiir_interface_library)


add_aiir_interface_library(AlignmentAttrInterface)
add_aiir_interface_library(CallInterfaces)
add_aiir_interface_library(CastInterfaces)
add_aiir_interface_library(ControlFlowInterfaces)
add_aiir_interface_library(DataLayoutInterfaces)
add_aiir_interface_library(DerivedAttributeOpInterface)
add_aiir_interface_library(DestinationStyleOpInterface)

add_aiir_library(AIIRFunctionInterfaces
  FunctionInterfaces.cpp
  FunctionImplementation.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRFunctionInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRIR
)

add_aiir_interface_library(IndexingMapOpInterface)
add_aiir_interface_library(InferIntRangeInterface)

add_aiir_library(AIIRInferStridedMetadataInterface
  InferStridedMetadataInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRInferStridedMetadataInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRInferIntRangeInterface
  AIIRIR
)

add_aiir_interface_library(InferTypeOpInterface)

add_aiir_library(AIIRLoopLikeInterface
  LoopLikeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
)

add_aiir_interface_library(MemOpInterfaces)
add_aiir_interface_library(MemorySlotInterfaces)
add_aiir_interface_library(ParallelCombiningOpInterface)
add_aiir_interface_library(RuntimeVerifiableOpInterface)
add_aiir_interface_library(ShapedOpInterfaces)
add_aiir_interface_library(SideEffectInterfaces)

add_aiir_library(AIIRSubsetOpInterface
  SubsetOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRDestinationStyleOpInterface
  AIIRSubsetOpInterfaceIncGen
  AIIRValueBoundsOpInterface

  LINK_LIBS PUBLIC
  AIIRDestinationStyleOpInterface
  AIIRIR
  AIIRValueBoundsOpInterface
  )

add_aiir_library(AIIRTilingInterface
  TilingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRTilingInterfaceIncGen
  AIIRDialectUtils

  LINK_LIBS PUBLIC
  AIIRIR
)

add_aiir_interface_library(VectorInterfaces)
add_aiir_interface_library(ViewLikeInterface)

add_aiir_library(AIIRValueBoundsOpInterface
  ValueBoundsOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRDestinationStyleOpInterface
  AIIRValueBoundsOpInterfaceIncGen
  AIIRViewLikeInterface

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRDestinationStyleOpInterface
  AIIRIR
  )

add_subdirectory(Utils)
