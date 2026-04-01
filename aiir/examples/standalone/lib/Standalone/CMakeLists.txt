add_aiir_dialect_library(AIIRStandalone
        StandaloneTypes.cpp
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandalonePasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        AIIRStandaloneOpsIncGen
        AIIRStandalonePassesIncGen

        LINK_LIBS PUBLIC
        AIIRIR
        AIIRInferTypeOpInterface
        AIIRFuncDialect
        )
