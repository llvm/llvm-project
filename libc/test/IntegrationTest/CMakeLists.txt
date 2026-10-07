set(arch_specific_deps)
if(LIBC_TARGET_ARCHITECTURE_IS_AARCH64 AND NOT LIBC_TARGET_OS_IS_BAREMETAL)
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
    libc.hdr.stdint_proxy
    libc.src.__support.OSUtil.osutil
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.string_view
    libc.src.__support.macros.properties.architectures
    ${arch_specific_deps}
)
