set(LLVM_LINK_COMPONENTS
  MC
  Support
  AArch64Desc
  )

if(BOLT_BUILT_STANDALONE)
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64/AArch64.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64)
  tablegen(LLVM AArch64GenInstrInfo.inc -gen-instr-info)
  tablegen(LLVM AArch64GenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM AArch64GenSystemOperands.inc -gen-searchable-tables)
  tablegen(LLVM AArch64GenSubtargetInfo.inc -gen-subtarget)

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
