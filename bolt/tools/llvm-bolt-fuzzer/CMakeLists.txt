set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  )

add_llvm_fuzzer(llvm-bolt-fuzzer
  llvm-bolt-fuzzer.cpp
  )
