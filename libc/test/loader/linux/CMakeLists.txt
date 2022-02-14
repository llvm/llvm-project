if(NOT (EXISTS ${LIBC_SOURCE_DIR}/loader/linux/${LIBC_TARGET_ARCHITECTURE}))
  message("Skipping loader tests for target architecture ${LIBC_TARGET_ARCHITECTURE}.")
  return()
endif()

add_header_library(
  loader_test
  HDRS
    loader_test.h
  DEPENDS
    libc.src.__support.OSUtil.osutil
)

add_loader_test(
  loader_args_test
  SRC
    args_test.cpp
  DEPENDS
    .loader_test
    libc.loader.linux.crt1
  ARGS
    1 2 3
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

add_loader_test(
  loader_no_envp_test
  SRC
    main_without_envp.cpp
  DEPENDS
    .loader_test
    libc.loader.linux.crt1
)

add_loader_test(
  loader_no_args_test
  SRC
    main_without_args.cpp
  DEPENDS
    .loader_test
    libc.loader.linux.crt1
)

# TODO: Disableing this test temporarily.
# add_loader_test(
#   getenv_test
#   SRC
#     getenv_test.cpp
#   DEPENDS
#     .loader_test
#     libc.loader.linux.crt1
#     libc.src.stdlib.getenv
#   ENV
#     FRANCE=Paris
#     GERMANY=Berlin
# )


# TODO: Disableing this test temporarily to investigate why gold fails to link
# and produce an executable for this test. Test works all fine with ld.bfd.
#add_loader_test(
#  loader_tls_test
#  SRC
#    tls_test.cpp
#  DEPENDS
#    libc.config.linux.app_h
#    libc.include.errno
#    libc.include.sys_mman
#    libc.loader.linux.crt1
#    libc.src.assert.__assert_fail
#    libc.src.errno.__errno_location
#    libc.src.sys.mman.mmap
#)
