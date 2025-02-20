add_clang_library(MLIRCIRInterfaces
  CIRFPTypeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRAttrsEnumsGen
  MLIRCIRFPTypeInterfaceIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )
