set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DWP
  ExecutionEngine
  MC
  Object
  Support
  )

set(TARGET_LINK_LIBRARIES
  LLVMBOLTCore
  LLVMBOLTPasses
  LLVMBOLTProfile
  LLVMBOLTRuntimeLibs
  LLVMBOLTUtils
  )

if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND TARGET_LINK_LIBRARIES LLVMBOLTTargetAArch64)
  set(BOLT_AArch64 On)
endif()

if ("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND TARGET_LINK_LIBRARIES LLVMBOLTTargetX86)
  set(BOLT_X64 On)
endif()

add_llvm_library(LLVMBOLTRewrite
  BinaryPassManager.cpp
  BoltDiff.cpp
  DWARFRewriter.cpp
  ExecutableFileMemoryManager.cpp
  MachORewriteInstance.cpp
  RewriteInstance.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  )

target_link_libraries(LLVMBOLTRewrite
  PRIVATE
  ${TARGET_LINK_LIBRARIES}
  )

if (DEFINED BOLT_AArch64)
  target_compile_definitions(LLVMBOLTRewrite PRIVATE AARCH64_AVAILABLE)
endif()

if (DEFINED BOLT_X64)
  target_compile_definitions(LLVMBOLTRewrite PRIVATE X86_AVAILABLE)
endif()
