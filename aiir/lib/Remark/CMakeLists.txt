add_aiir_library(AIIRRemarkStreamer
  RemarkStreamer.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Remark

  LINK_LIBS PUBLIC
    AIIRIR

  LINK_COMPONENTS
    Remarks
    Core
    BitstreamReader
  )
