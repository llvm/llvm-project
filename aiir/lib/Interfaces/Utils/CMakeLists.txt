set(LLVM_OPTIONAL_SOURCES
  InferIntRangeCommon.cpp
  MemorySlotUtils.cpp
  )

add_aiir_library(AIIRInferIntRangeCommon
    InferIntRangeCommon.cpp

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces/Utils

    DEPENDS
    AIIRInferIntRangeInterfaceIncGen

    LINK_LIBS PUBLIC
    AIIRShapedOpInterfaces
    AIIRInferIntRangeInterface
    AIIRIR
)

add_aiir_library(AIIRMemorySlotUtils
    MemorySlotUtils.cpp

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces/Utils

    LINK_LIBS PUBLIC
    AIIRIR
)
