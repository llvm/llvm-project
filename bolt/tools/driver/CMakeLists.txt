set(LLVM_LINK_COMPONENTS
  ${BOLT_TARGETS_TO_BUILD}
  MC
  Object
  Support
  )

add_bolt_tool(llvm-bolt llvm-bolt.cpp DISABLE_LLVM_LINK_LLVM_DYLIB)

target_link_libraries(llvm-bolt
  PRIVATE
  LLVMBOLTProfile
  LLVMBOLTRewrite
  LLVMBOLTUtils
  )

add_bolt_tool_symlink(perf2bolt llvm-bolt)
add_bolt_tool_symlink(llvm-boltdiff llvm-bolt)

add_dependencies(bolt llvm-bolt)

include_directories(${BOLT_SOURCE_DIR}/lib)
