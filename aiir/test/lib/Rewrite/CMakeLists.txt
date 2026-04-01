if (AIIR_ENABLE_PDL_IN_PATTERNMATCH)
  # Exclude tests from libAIIR.so
  add_aiir_library(AIIRTestRewrite
    TestPDLByteCode.cpp

    EXCLUDE_FROM_LIBAIIR

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Rewrite
    )
  aiir_target_link_libraries(AIIRTestRewrite PUBLIC
    AIIRIR
    AIIRPass
    AIIRSupport
    AIIRTransformUtils
    )
endif()
