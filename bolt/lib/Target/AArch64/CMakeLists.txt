set(LLVM_LINK_COMPONENTS
  MC
  Support
  AArch64Desc
  )

add_llvm_library(LLVMBOLTTargetAArch64
  AArch64MCPlusBuilder.cpp

  DEPENDS
  AArch64CommonTableGen
  )

target_link_libraries(LLVMBOLTTargetAArch64 PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_BINARY_DIR}/lib/Target/AArch64
  )
