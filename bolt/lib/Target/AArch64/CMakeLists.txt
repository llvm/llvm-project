set(LLVM_LINK_COMPONENTS
  MC
  Support
  AArch64Desc
  )

if(BOLT_BUILT_STANDALONE)
  # tablegen, copied from llvm/lib/Target/AAarch64/CMakeLists.txt
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64/AArch64.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64)
  tablegen(LLVM AArch64GenAsmMatcher.inc -gen-asm-matcher)
  tablegen(LLVM AArch64GenAsmWriter.inc -gen-asm-writer)
  tablegen(LLVM AArch64GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
  tablegen(LLVM AArch64GenCallingConv.inc -gen-callingconv)
  tablegen(LLVM AArch64GenDAGISel.inc -gen-dag-isel)
  tablegen(LLVM AArch64GenDisassemblerTables.inc -gen-disassembler)
  tablegen(LLVM AArch64GenFastISel.inc -gen-fast-isel)
  tablegen(LLVM AArch64GenGlobalISel.inc -gen-global-isel)
  tablegen(LLVM AArch64GenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="AArch64O0PreLegalizerCombiner")
  tablegen(LLVM AArch64GenPreLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="AArch64PreLegalizerCombiner")
  tablegen(LLVM AArch64GenPostLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="AArch64PostLegalizerCombiner")
  tablegen(LLVM AArch64GenPostLegalizeGILowering.inc -gen-global-isel-combiner
                -combiners="AArch64PostLegalizerLowering")
  tablegen(LLVM AArch64GenInstrInfo.inc -gen-instr-info)
  tablegen(LLVM AArch64GenMCCodeEmitter.inc -gen-emitter)
  tablegen(LLVM AArch64GenMCPseudoLowering.inc -gen-pseudo-lowering)
  tablegen(LLVM AArch64GenRegisterBank.inc -gen-register-bank)
  tablegen(LLVM AArch64GenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM AArch64GenSubtargetInfo.inc -gen-subtarget)
  tablegen(LLVM AArch64GenSystemOperands.inc -gen-searchable-tables)
  tablegen(LLVM AArch64GenExegesis.inc -gen-exegesis)

  add_public_tablegen_target(AArch64CommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

add_llvm_library(LLVMBOLTTargetAArch64
  AArch64MCPlusBuilder.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  AArch64CommonTableGen
  )

target_link_libraries(LLVMBOLTTargetAArch64 PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_BINARY_DIR}/lib/Target/AArch64
  )
