set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Demangle
  MC
  MCDisassembler
  Object
  Support
  )

add_llvm_library(LLVMBOLTCore
  BinaryBasicBlock.cpp
  BinaryContext.cpp
  BinaryData.cpp
  BinaryEmitter.cpp
  BinaryFunction.cpp
  BinaryFunctionProfile.cpp
  BinarySection.cpp
  DebugData.cpp
  DynoStats.cpp
  Exceptions.cpp
  JumpTable.cpp
  MCPlusBuilder.cpp
  ParallelUtilities.cpp
  Relocation.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  )

target_link_libraries(LLVMBOLTCore
  PRIVATE
  LLVMBOLTUtils
  )
