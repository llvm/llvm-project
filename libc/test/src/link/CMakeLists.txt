add_custom_target(libc_link_unittests)

add_libc_test(
  dl_iterate_phdr_test
  SUITE
    libc_link_unittests
  SRCS
    dl_iterate_phdr_test.cpp
  DEPENDS
    libc.src.link.dl_iterate_phdr
)
