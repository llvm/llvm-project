set(LLVM_LINK_COMPONENTS
  BOLTCore
  MC
  Object
  Support
  ${LLVM_TARGETS_TO_BUILD}
  )

add_llvm_library(LLVMBOLTTargetX86
  X86MCPlusBuilder.cpp

  DEPENDS
  intrinsics_gen
  X86CommonTableGen
  )

include_directories(${LLVM_MAIN_SRC_DIR}/lib/Target/X86 ${LLVM_BINARY_DIR}/lib/Target/X86)
