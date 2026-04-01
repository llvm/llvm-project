set(LLVM_OPTIONAL_SOURCES
  MemorySpaceInterfaces.cpp
  PtrAttrs.cpp
  PtrTypes.cpp
  PtrDialect.cpp
)

add_aiir_dialect_library(
  AIIRPtrMemorySpaceInterfaces
  MemorySpaceInterfaces.cpp

  DEPENDS
  AIIRPtrOpsEnumsGen
  AIIRPtrMemorySpaceInterfacesIncGen
  LINK_LIBS
  PUBLIC
  AIIRIR
)

add_aiir_dialect_library(
  AIIRPtrDialect
  PtrAttrs.cpp
  PtrTypes.cpp
  PtrDialect.cpp

  DEPENDS
  AIIRPtrOpsAttributesIncGen
  AIIRPtrOpsIncGen
  AIIRPtrOpsEnumsGen
  AIIRPtrMemorySpaceInterfacesIncGen
  LINK_LIBS
  PUBLIC
  AIIRIR
  AIIRDataLayoutInterfaces
  AIIRMemorySlotInterfaces
  AIIRInferTypeOpInterface
  AIIRViewLikeInterface
  AIIRPtrMemorySpaceInterfaces
)
