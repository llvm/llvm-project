add_aiir_library(AIIRSPIRVImageInterfaces
  SPIRVImageInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV

  DEPENDS
  AIIRSPIRVImageInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRIR
)
