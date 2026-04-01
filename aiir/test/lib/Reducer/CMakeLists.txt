# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestReducer
  AIIRTestReducer.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  LINK_COMPONENTS
  Core
  )
aiir_target_link_libraries(AIIRTestReducer PUBLIC
  AIIRIR
  AIIRPass
  AIIRSupport
  )
