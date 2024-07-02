set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  X86Desc
  )

if(BOLT_BUILT_STANDALONE)
  # tablegen, copied from llvm/lib/Target/X86/CMakeLists.txt
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/X86/X86.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/X86)
  tablegen(LLVM X86GenAsmMatcher.inc -gen-asm-matcher)
  tablegen(LLVM X86GenAsmWriter.inc -gen-asm-writer)
  tablegen(LLVM X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
  tablegen(LLVM X86GenCallingConv.inc -gen-callingconv)
  tablegen(LLVM X86GenDAGISel.inc -gen-dag-isel)
  tablegen(LLVM X86GenDisassemblerTables.inc -gen-disassembler)
  tablegen(LLVM X86GenCompressEVEXTables.inc -gen-x86-compress-evex-tables)
  tablegen(LLVM X86GenExegesis.inc -gen-exegesis)
  tablegen(LLVM X86GenFastISel.inc -gen-fast-isel)
  tablegen(LLVM X86GenGlobalISel.inc -gen-global-isel)
  tablegen(LLVM X86GenInstrInfo.inc -gen-instr-info
                                    -instr-info-expand-mi-operand-info=0)
  tablegen(LLVM X86GenMnemonicTables.inc -gen-x86-mnemonic-tables -asmwriternum=1)
  tablegen(LLVM X86GenRegisterBank.inc -gen-register-bank)
  tablegen(LLVM X86GenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM X86GenSubtargetInfo.inc -gen-subtarget)
  tablegen(LLVM X86GenFoldTables.inc -gen-x86-fold-tables -asmwriternum=1)

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
