add_aiir_library(AIIRPass
  IRPrinting.cpp
  Pass.cpp
  PassCrashRecovery.cpp
  PassManagerOptions.cpp
  PassRegistry.cpp
  PassStatistics.cpp
  PassTiming.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Pass

  DEPENDS
  aiir-generic-headers

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRIR
  )
