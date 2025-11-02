set(LLVM_LINK_COMPONENTS
  Core
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  JITLink
  MC
  Object
  Support
  DWARFLinker
  DWARFLinkerClassic
  AsmPrinter
  TargetParser
  )

add_llvm_library(LLVMBOLTRewrite
  BinaryPassManager.cpp
  BoltDiff.cpp
  DWARFRewriter.cpp
  ExecutableFileMemoryManager.cpp
  JITLinkLinker.cpp
  LinuxKernelRewriter.cpp
  MachORewriteInstance.cpp
  MetadataManager.cpp
  BuildIDRewriter.cpp
  PseudoProbeRewriter.cpp
  RewriteInstance.cpp
  SDTRewriter.cpp
  GNUPropertyRewriter.cpp

  NO_EXPORT
  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  )

target_link_libraries(LLVMBOLTRewrite
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTPasses
  LLVMBOLTProfile
  LLVMBOLTRuntimeLibs
  LLVMBOLTUtils
  )

foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  target_link_libraries(LLVMBOLTRewrite PRIVATE LLVMBOLTTarget${tgt})
  string(TOUPPER "${tgt}" upper)
  target_compile_definitions(LLVMBOLTRewrite PRIVATE ${upper}_AVAILABLE)
endforeach()
