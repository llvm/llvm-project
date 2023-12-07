set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Object
  Support
  )

if (BOLT_ENABLE_RUNTIME)
  set(BOLT_DRIVER_DEPS "bolt_rt")
else()
  set(BOLT_DRIVER_DEPS "")
endif()

add_bolt_tool(llvm-bolt
  llvm-bolt.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  ${BOLT_DRIVER_DEPS}
  )

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
