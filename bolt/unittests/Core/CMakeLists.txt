set(LLVM_LINK_COMPONENTS
  BOLTCore
  BOLTRewrite
  DebugInfoDWARF
  Object
  MC
  ${LLVM_TARGETS_TO_BUILD}
  )

add_bolt_unittest(CoreTests
  MCPlusBuilder.cpp
  )

if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  include_directories(
    ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
    ${LLVM_BINARY_DIR}/lib/Target/AArch64
  )

  target_compile_definitions(CoreTests PRIVATE AARCH64_AVAILABLE)
endif()

if ("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  include_directories(
    ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
    ${LLVM_BINARY_DIR}/lib/Target/X86
  )

  target_compile_definitions(CoreTests PRIVATE X86_AVAILABLE)
endif()
