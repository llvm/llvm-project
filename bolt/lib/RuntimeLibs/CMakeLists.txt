set(LLVM_LINK_COMPONENTS
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

target_link_libraries(LLVMBOLTRuntimeLibs
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTPasses
  LLVMBOLTUtils
  )
