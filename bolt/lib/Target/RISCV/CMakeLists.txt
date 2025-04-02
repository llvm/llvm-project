set(LLVM_LINK_COMPONENTS
  MC
  Support
  RISCVDesc
  )

if(BOLT_BUILT_STANDALONE)
  # tablegen, copied from llvm/lib/Target/RISCV/CMakeLists.txt
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV/RISCV.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV)
  tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
  tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
  tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)

  add_public_tablegen_target(RISCVCommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

add_llvm_library(LLVMBOLTTargetRISCV
  RISCVMCPlusBuilder.cpp

  NO_EXPORT
  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  RISCVCommonTableGen
  )

target_link_libraries(LLVMBOLTTargetRISCV PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  )
