if(NOT TARGET libc.startup.linux.gnu_property_section)
  return()
endif()

add_libc_testsuite(libc-startup-unit-tests)

add_libc_test(
  gnu_property_section_test
  FULL_BUILD_ONLY
  SUITE
    libc-startup-unit-tests
  SRCS
    gnu_property_section_test.cpp
  DEPENDS
    libc.hdr.elf_proxy
    libc.hdr.link_macros
    libc.hdr.stdint_proxy
    libc.src.__support.macros.properties.architectures
    libc.startup.linux.gnu_property_section
)
