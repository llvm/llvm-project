add_aiir_pdll_library(AIIRTestPDLLPatternsIncGen
  TestPDLL.pdll
  TestPDLLPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )

# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestPDLL
  TestPDLL.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/PDLL

  DEPENDS
  AIIRTestPDLLPatternsIncGen

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestPDLL PUBLIC
  AIIRCastInterfaces
  AIIRIR
  AIIRPass
  AIIRPDLInterpDialect
  AIIRPDLDialect
  AIIRSupport
  AIIRTransformUtils
  )

target_include_directories(AIIRTestPDLL
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
