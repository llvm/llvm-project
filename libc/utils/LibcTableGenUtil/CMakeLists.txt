add_llvm_library(
  LibcTableGenUtil
  APIIndexer.cpp
  APIIndexer.h
  DISABLE_LLVM_LINK_LLVM_DYLIB
  LINK_COMPONENTS Support TableGen
)
target_include_directories(LibcTableGenUtil PUBLIC ${LIBC_SOURCE_DIR})
target_include_directories(LibcTableGenUtil PRIVATE ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
