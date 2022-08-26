set(LLVM_LINK_COMPONENTS
  BOLTCore
  BOLTUtils
  MC
  MCDisassembler
  Support
  X86Desc
  )

add_llvm_library(LLVMBOLTTargetX86
  X86MCPlusBuilder.cpp
  X86MCSymbolizer.cpp

  DEPENDS
  X86CommonTableGen
  )

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_LIBRARY_DIR}/Target/X86
  )
