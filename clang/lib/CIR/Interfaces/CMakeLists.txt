add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIREnumsGen
  mlir-headers

  LINK_LIBS
  MLIRIR
  MLIRPtrMemorySpaceInterfaces
  MLIRSupport
 )
