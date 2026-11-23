add_libc_test(
  error_test
  SUITE libc-osutil-tests
  SRCS error_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.include.llvm-libc-types.EFI_STATUS
    libc.src.__support.CPP.span
    libc.src.__support.OSUtil.uefi.uefi_error
)
