add_llvm_library(
  LibcTableGenUtil
  APIIndexer.cpp
  APIIndexer.h
  LINK_COMPONENTS Support
)
target_include_directories(LibcTableGenUtil PUBLIC ${LIBC_SOURCE_DIR})
