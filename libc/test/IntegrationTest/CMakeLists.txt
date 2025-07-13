set(arch_specific_deps)
if(LIBC_TARGET_ARCHITECTURE_IS_AARCH64)
  set(arch_specific_deps libc.src.sys.auxv.getauxval)
endif()
add_object_library(
  test
  SRCS
    test.cpp
  COMPILE_OPTIONS
    ${TEST_COMPILE_FLAGS}
  HDRS
    test.h
  DEPENDS
    libc.src.__support.OSUtil.osutil
    ${arch_specific_deps}
)
