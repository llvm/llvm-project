set(LLVM_LINK_COMPONENTS
  BinaryFormat
  JITLink
  MC
  Object
  Support
  )

add_llvm_library(LLVMBOLTRuntimeLibs
  RuntimeLibrary.cpp
  HugifyRuntimeLibrary.cpp
  InstrumentationRuntimeLibrary.cpp

  NO_EXPORT
  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(LLVMBOLTRuntimeLibs
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTPasses
  LLVMBOLTUtils
  )
