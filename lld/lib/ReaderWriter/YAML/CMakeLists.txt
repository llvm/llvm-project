add_lld_library(lldYAML
  ReaderWriterYAML.cpp
  LINK_LIBS
    lldCore
    LLVMSupport
  )
