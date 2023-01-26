set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  X86Desc
  )

add_llvm_library(LLVMBOLTTargetX86
  X86MCPlusBuilder.cpp
  X86MCSymbolizer.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  X86CommonTableGen
  )

target_link_libraries(LLVMBOLTTargetX86 PRIVATE LLVMBOLTCore LLVMBOLTUtils)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )
