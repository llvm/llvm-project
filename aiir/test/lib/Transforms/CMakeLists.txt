set(LLVM_TARGET_DEFINITIONS TestTransformsOps.td)
aiir_tablegen(TestTransformsOps.h.inc -gen-op-decls)
aiir_tablegen(TestTransformsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIIRTestTransformsOpsIncGen)

set(LLVM_OPTIONAL_SOURCES 
  TestDialectConversion.cpp)
set(AIIRTestTransformsPDLDep)
set(AIIRTestTransformsPDLSrc)
if(AIIR_ENABLE_PDL_IN_PATTERNMATCH)
add_aiir_pdll_library(AIIRTestDialectConversionPDLLPatternsIncGen
  TestDialectConversion.pdll
  TestDialectConversionPDLLPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
  set(AIIRTestTransformsPDLSrc
    TestDialectConversion.cpp)
  set(AIIRTestTransformsPDLDep
    AIIRTestDialectConversionPDLLPatternsIncGen)
endif()

# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestTransforms
  TestCommutativityUtils.cpp
  TestCompositePass.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestInliningCallback.cpp
  TestMakeIsolatedFromAbove.cpp
  TestSingleFold.cpp
  TestTransformsOps.cpp
  ${AIIRTestTransformsPDLSrc}

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Transforms

  DEPENDS
  ${AIIRTestTransformsPDLDep}
  AIIRTestTransformsOpsIncGen

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestTransforms PUBLIC
  AIIRAnalysis
  AIIRFuncDialect
  AIIRInferIntRangeInterface
  AIIRTransforms
  AIIRTransformDialect
  )

target_include_directories(AIIRTestTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
