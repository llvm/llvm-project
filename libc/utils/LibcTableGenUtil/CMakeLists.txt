add_llvm_library(
  LibcTableGenUtil
  APIIndexer.cpp
  APIIndexer.h
  LINK_COMPONENTS Support TableGen
)
target_include_directories(LibcTableGenUtil PUBLIC ${LIBC_SOURCE_DIR} ${LLVM_LIBC_INCLUDE_DIRS})
