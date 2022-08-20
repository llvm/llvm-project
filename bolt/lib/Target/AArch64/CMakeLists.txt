set(LLVM_LINK_COMPONENTS
  BOLTCore
  MC
  Support
  AArch64Desc
  )

add_llvm_library(LLVMBOLTTargetAArch64
  AArch64MCPlusBuilder.cpp

  DEPENDS
  AArch64CommonTableGen
  )

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_LIBRARY_DIR}/Target/AArch64
  )
