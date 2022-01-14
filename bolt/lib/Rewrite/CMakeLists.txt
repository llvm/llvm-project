set(LLVM_LINK_COMPONENTS
  BOLTCore
  BOLTPasses
  BOLTProfile
  BOLTRuntimeLibs
  BOLTUtils
  DebugInfoDWARF
  DWP
  ExecutionEngine
  MC
  Object
  Support
  )

if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND LLVM_LINK_COMPONENTS BOLTTargetAArch64)
  set(BOLT_AArch64 On)
endif()

if ("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND LLVM_LINK_COMPONENTS BOLTTargetX86)
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

if (DEFINED BOLT_AArch64)
  target_compile_definitions(LLVMBOLTRewrite PRIVATE AARCH64_AVAILABLE)
endif()

if (DEFINED BOLT_X64)
  target_compile_definitions(LLVMBOLTRewrite PRIVATE X86_AVAILABLE)
endif()
