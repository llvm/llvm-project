add_aiir_dialect_library(AIIRWasmSSADialect
  WasmSSAOps.cpp
  WasmSSADialect.cpp
  WasmSSAInterfaces.cpp
  WasmSSATypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/WasmSSA

  DEPENDS
  AIIRWasmSSAOpsIncGen
  AIIRWasmSSAInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRCastInterfaces
  AIIRDataLayoutInterfaces
  AIIRDialect
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRSupport

  PRIVATE
  AIIRFunctionInterfaces
  )
