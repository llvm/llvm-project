add_clang_library(MLIRCIRInterfaces
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRFPTypeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIREnumsGen
  MLIRCIRFPTypeInterfaceIncGen
  MLIRCIRLoopOpInterfaceIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )
