add_llvm_library(LLVMBOLTUtils
  CommandLineOpts.cpp
  Utils.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  Support
  )
