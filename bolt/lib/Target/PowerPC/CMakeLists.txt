set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  PowerPCDesc
)

if(BOLT_BUILT_STANDALONE)
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC/PPC.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC)
  tablegen(LLVM PPCGenInstrInfo.inc      -gen-instr-info)
  tablegen(LLVM PPCGenRegisterInfo.inc   -gen-register-info)
  tablegen(LLVM PPCGenSubtargetInfo.inc  -gen-subtarget)
  add_public_tablegen_target(PowerPCCommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

add_llvm_library(LLVMBOLTTargetPowerPC
  PPCMCPlusBuilder.cpp

  NO_EXPORT
  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  PowerPCCommonTableGen
)

target_link_libraries(LLVMBOLTTargetPowerPC PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC
  ${LLVM_BINARY_DIR}/lib/Target/PowerPC
)
