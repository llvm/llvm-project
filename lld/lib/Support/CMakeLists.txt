add_lld_library(lldSupport
  Memory.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Support

  LINK_LIBS
  LLVMSupport
)
