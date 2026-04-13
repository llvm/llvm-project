set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Object
  MC
  ${BOLT_TARGETS_TO_BUILD}
  )

add_bolt_unittest(CoreTests
  BinaryContext.cpp
  ClusteredRows.cpp
  MCPlusBuilder.cpp
  MemoryMaps.cpp
  DynoStats.cpp

  # FIXME CoreTests uses `llvm::detail::TakeError(llvm::Error)`, but linking
  #       to LLVMTestingSupport introduces a transitive dependency on the
  #       dynamic LLVM library when LLVM_LINK_LLVM_DYLIB is ON.
  ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support/Error.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(CoreTests
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTRewrite
  LLVMBOLTProfile
  LLVMBOLTUtils
  )

foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  include_directories(
    ${LLVM_MAIN_SRC_DIR}/lib/Target/${tgt}
    ${LLVM_BINARY_DIR}/lib/Target/${tgt}
  )
  string(TOUPPER "${tgt}" upper)
  target_compile_definitions(CoreTests PRIVATE "${upper}_AVAILABLE")
endforeach()
