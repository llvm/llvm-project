add_clang_library(AIIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRCIRASTAttrInterfacesIncGen
  AIIRCIREnumsGen
  AIIRCIRTypeInterfacesIncGen
  AIIRCIRLoopOpInterfaceIncGen
  AIIRCIROpInterfacesIncGen

  LINK_LIBS
  ${dialect_libs}
  AIIRIR
  AIIRSupport
 )
