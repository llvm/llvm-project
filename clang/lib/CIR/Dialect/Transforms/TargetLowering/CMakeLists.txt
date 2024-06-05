add_clang_library(TargetLowering
  ABIInfo.cpp
  ABIInfoImpl.cpp
  CIRCXXABI.cpp
  CIRLowerContext.cpp
  CIRRecordLayout.cpp
  ItaniumCXXABI.cpp
  LowerCall.cpp
  LowerFunction.cpp
  LowerModule.cpp
  LowerTypes.cpp
  RecordLayoutBuilder.cpp
  TargetInfo.cpp
  TargetLoweringInfo.cpp
  Targets/X86.cpp
  Targets/LoweringPrepareAArch64CXXABI.cpp
  Targets/LoweringPrepareItaniumCXXABI.cpp

  DEPENDS

  LINK_LIBS PUBLIC

  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
