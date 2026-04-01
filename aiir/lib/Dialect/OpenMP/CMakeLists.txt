add_subdirectory(Transforms)

add_aiir_dialect_library(AIIROpenMPDialect
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenMP

  DEPENDS
  omp_gen
  AIIROpenMPOpsIncGen
  AIIROpenMPOpsInterfacesIncGen
  AIIROpenMPTypeInterfacesIncGen

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRFuncDialect
  AIIROpenACCMPCommon
  )
