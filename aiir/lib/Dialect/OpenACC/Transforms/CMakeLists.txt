add_aiir_dialect_library(AIIROpenACCTransforms
  ACCBindRoutine.cpp
  ACCComputeLowering.cpp
  ACCRoutineLowering.cpp
  ACCRoutineToGPUFunc.cpp
  ACCDeclareGPUModuleInsertion.cpp
  ACCIfClauseLowering.cpp
  ACCImplicitData.cpp
  ACCRecipeMaterialization.cpp
  ACCLoopTiling.cpp
  ACCImplicitDeclare.cpp
  ACCImplicitRoutine.cpp
  ACCLegalizeSerial.cpp
  ACCSpecializeForDevice.cpp
  ACCSpecializeForHost.cpp
  LegalizeDataValues.cpp
  OffloadLiveInValueCanonicalization.cpp
  OffloadTargetVerifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenACC

  DEPENDS
  AIIROpenACCPassIncGen
  AIIROpenACCOpsIncGen
  AIIROpenACCEnumsIncGen
  AIIROpenACCAttributesIncGen
  AIIROpenACCMPOpsInterfacesIncGen
  AIIROpenACCOpsInterfacesIncGen
  AIIROpenACCTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRArithDialect
  AIIRArithUtils
  AIIROpenACCAnalysis
  AIIROpenACCDialect
  AIIROpenACCUtils
  AIIRFuncDialect
  AIIRIR
  AIIRPass
  AIIRSCFDialect
  AIIRSupport
  AIIRTransforms
)
