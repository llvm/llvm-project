set(LLVM_LINK_COMPONENTS
  BOLTRewrite
  )

add_bolt_unittest(CoreTests
  MCPlusBuilder.cpp
  )

string(FIND "${LLVM_TARGETS_TO_BUILD}" "AArch64" POSITION)
if (NOT ${POSITION} EQUAL -1)
  include_directories(
    ${CMAKE_SOURCE_DIR}/lib/Target/AArch64
    ${CMAKE_BINARY_DIR}/lib/Target/AArch64
  )

  target_compile_definitions(CoreTests PRIVATE AARCH64_AVAILABLE)
endif()

string(FIND "${LLVM_TARGETS_TO_BUILD}" "X86" POSITION)
if (NOT ${POSITION} EQUAL -1)
  include_directories(
    ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
    ${LLVM_BINARY_DIR}/lib/Target/X86
  )

  target_compile_definitions(CoreTests PRIVATE X86_AVAILABLE)
endif()
