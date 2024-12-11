set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  X86Desc
  )

if(BOLT_BUILT_STANDALONE)
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/X86/X86.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/X86)
  tablegen(LLVM X86GenInstrInfo.inc -gen-instr-info -instr-info-expand-mi-operand-info=0)
  tablegen(LLVM X86GenMnemonicTables.inc -gen-x86-mnemonic-tables -asmwriternum=1)
  tablegen(LLVM X86GenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM X86GenSubtargetInfo.inc -gen-subtarget)

  add_public_tablegen_target(X86CommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

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
