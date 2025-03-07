add_clang_library(MLIRCIRInterfaces
  CIROpInterfaces.cpp
  CIRFPTypeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIREnumsGen
  MLIRCIRFPTypeInterfaceIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )
