set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  )

add_llvm_fuzzer(llvm-bolt-fuzzer
  llvm-bolt-fuzzer.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

if (TARGET llvm-bolt-fuzzer)
    target_link_libraries(llvm-bolt-fuzzer PRIVATE LLVMBOLTRewrite)
endif()
