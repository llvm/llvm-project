set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Object
  ${LLVM_TARGETS_TO_BUILD}
  )

add_bolt_unittest(ProfileTests
  DataAggregator.cpp
  PerfSpeEvents.cpp
  PerfScripts.cpp

  # FIXME CoreTests uses `llvm::detail::TakeError(llvm::Error)`, but linking
  #       to LLVMTestingSupport introduces a transitive dependency on the
  #       dynamic LLVM library when LLVM_LINK_LLVM_DYLIB is ON.
  ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support/Error.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(ProfileTests
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTProfile
  LLVMTargetParser
  )

foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  string(TOUPPER "${tgt}" upper)
  target_compile_definitions(ProfileTests PRIVATE "${upper}_AVAILABLE")
endforeach()
