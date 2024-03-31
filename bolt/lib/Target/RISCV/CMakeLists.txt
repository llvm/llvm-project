set(LLVM_LINK_COMPONENTS
  MC
  Support
  RISCVDesc
  )

if(BOLT_BUILT_STANDALONE)
  # tablegen, copied from llvm/lib/Target/RISCV/CMakeLists.txt
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV/RISCV.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV)
  tablegen(LLVM RISCVGenAsmMatcher.inc -gen-asm-matcher)
  tablegen(LLVM RISCVGenAsmWriter.inc -gen-asm-writer)
  tablegen(LLVM RISCVGenCompressInstEmitter.inc -gen-compress-inst-emitter)
  tablegen(LLVM RISCVGenMacroFusion.inc -gen-macro-fusion-pred)
  tablegen(LLVM RISCVGenDAGISel.inc -gen-dag-isel)
  tablegen(LLVM RISCVGenDisassemblerTables.inc -gen-disassembler)
  tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
  tablegen(LLVM RISCVGenMCCodeEmitter.inc -gen-emitter)
  tablegen(LLVM RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
  tablegen(LLVM RISCVGenRegisterBank.inc -gen-register-bank)
  tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
  tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)

  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV/RISCVGISel.td)
  tablegen(LLVM RISCVGenGlobalISel.inc -gen-global-isel)
  tablegen(LLVM RISCVGenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="RISCVO0PreLegalizerCombiner")
  tablegen(LLVM RISCVGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="RISCVPreLegalizerCombiner")
  tablegen(LLVM RISCVGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
                -combiners="RISCVPostLegalizerCombiner")

  add_public_tablegen_target(RISCVCommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

add_llvm_library(LLVMBOLTTargetRISCV
  RISCVMCPlusBuilder.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  RISCVCommonTableGen
  )

target_link_libraries(LLVMBOLTTargetRISCV PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  )
