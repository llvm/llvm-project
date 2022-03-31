add_llvm_library(LLVMBOLTProfile
  BoltAddressTranslation.cpp
  DataAggregator.cpp
  DataReader.cpp
  Heatmap.cpp
  ProfileReaderBase.cpp
  YAMLProfileReader.cpp
  YAMLProfileWriter.cpp

  LINK_COMPONENTS
  BOLTCore
  BOLTPasses
  BOLTUtils
  Support
  )
