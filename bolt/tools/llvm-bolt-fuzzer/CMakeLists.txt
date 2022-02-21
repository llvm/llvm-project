set(LLVM_LINK_COMPONENTS
  BOLTRewrite
  )

add_llvm_fuzzer(llvm-bolt-fuzzer
  llvm-bolt-fuzzer.cpp
  )
