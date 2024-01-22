add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )