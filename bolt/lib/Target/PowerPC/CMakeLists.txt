add_llvm_library(LLVMBOLTTargetPowerPC
    PPCMCPlusBuilder.cpp
  )

target_include_directories(LLVMBOLTTargetPowerPC PRIVATE
    ${LLVM_BINARY_DIR}/include
    ${LLVM_SOURCE_DIR}/include
)