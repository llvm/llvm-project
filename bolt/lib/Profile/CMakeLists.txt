add_llvm_library(LLVMBOLTProfile
  BoltAddressTranslation.cpp
  DataAggregator.cpp
  DataReader.cpp
  Heatmap.cpp
  StaleProfileMatching.cpp
  YAMLProfileReader.cpp
  YAMLProfileWriter.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  Demangle
  MC
  Support
  TransformUtils
  )

target_link_libraries(LLVMBOLTProfile
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTUtils
  )
