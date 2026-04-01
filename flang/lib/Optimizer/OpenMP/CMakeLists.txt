get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  AutomapToTargetData.cpp
  DoConcurrentConversion.cpp
  FunctionFiltering.cpp
  GenericLoopConversion.cpp
  MapsForPrivatizedSymbols.cpp
  MapInfoFinalization.cpp
  DeleteUnreachableTargets.cpp
  LowerWorkdistribute.cpp
  LowerWorkshare.cpp
  LowerNontemporal.cpp
  SimdOnly.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  FlangOpenMPPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranSupport
  HLFIRDialect
  FortranUtils

  AIIR_DEPS
  ${dialect_libs}

  AIIR_LIBS
  AIIRFuncDialect
  AIIROpenMPDialect
  AIIROpenMPTransforms
  AIIRIR
  AIIRPass
  AIIRTransformUtils
  ${dialect_libs}
)

add_subdirectory(Support)
