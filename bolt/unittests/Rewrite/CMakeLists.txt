set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Object
  MC
  ${LLVM_TARGETS_TO_BUILD}
  )

add_bolt_unittest(BOLTRewriteTests
  JITRewriteInstance.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(BOLTRewriteTests
  PRIVATE
  LLVMBOLTCore
  LLVMBOLTRewrite
  )

foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  include_directories(
    ${LLVM_MAIN_SRC_DIR}/lib/Target/${tgt}
    ${LLVM_BINARY_DIR}/lib/Target/${tgt}
  )
  string(TOUPPER "${tgt}" upper)
  target_compile_definitions(BOLTRewriteTests PRIVATE "${upper}_AVAILABLE")
endforeach()
