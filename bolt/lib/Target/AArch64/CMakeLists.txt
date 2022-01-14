set(LLVM_LINK_COMPONENTS
  BOLTCore
  MC
  Object
  Support
  ${LLVM_TARGETS_TO_BUILD}
  )

add_llvm_library(LLVMBOLTTargetAArch64
  AArch64MCPlusBuilder.cpp

  DEPENDS
  intrinsics_gen
  AArch64CommonTableGen
  )

include_directories(${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64 ${LLVM_BINARY_DIR}/lib/Target/AArch64)
