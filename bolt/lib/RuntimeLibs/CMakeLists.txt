set(LLVM_LINK_COMPONENTS
  BOLTCore
  BOLTPasses
  BOLTUtils
  BinaryFormat
  MC
  Object
  RuntimeDyld
  Support
  )

add_llvm_library(LLVMBOLTRuntimeLibs
  RuntimeLibrary.cpp
  HugifyRuntimeLibrary.cpp
  InstrumentationRuntimeLibrary.cpp
  )
