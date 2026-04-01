set(LLVM_OPTIONAL_SOURCES ByteCode.cpp)

add_aiir_library(AIIRRewrite
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Rewrite

  DEPENDS
  aiir-generic-headers
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSideEffectInterfaces
  )

if(AIIR_ENABLE_PDL_IN_PATTERNMATCH)
  add_aiir_library(AIIRRewritePDL
    ByteCode.cpp

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Rewrite

    DEPENDS
    aiir-generic-headers

    LINK_LIBS PUBLIC
    AIIRIR
    AIIRPDLDialect
    AIIRPDLInterpDialect
    AIIRPDLToPDLInterp
    AIIRSideEffectInterfaces
  )

  target_link_libraries(AIIRRewrite PUBLIC
    AIIRPDLDialect
    AIIRPDLInterpDialect
    AIIRPDLToPDLInterp
    AIIRRewritePDL)
endif()

