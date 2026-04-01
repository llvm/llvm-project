add_llvm_library(StandalonePlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        standalone-plugin.cpp

        DEPENDS
        AIIRStandalone
        PLUGIN_TOOL
        aiir-opt

        LINK_LIBS
        AIIRStandalone
        )
target_include_directories(
        StandalonePlugin
        PRIVATE
        "${STANDALONE_BINARY_DIR}/include"
)
