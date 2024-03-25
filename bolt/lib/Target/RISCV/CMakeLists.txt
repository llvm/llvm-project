set(LLVM_LINK_COMPONENTS
  MC
  Support
  RISCVDesc
  )

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
