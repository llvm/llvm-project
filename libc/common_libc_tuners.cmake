# ------------------------------------------------------------------------------
# Common tuning option definitions.
# ------------------------------------------------------------------------------

set(LIBC_COMMON_TUNE_OPTIONS "")

option(LIBC_UNSAFE_STRING_WIDE_READ "Functions searching for the first character in a string such as strlen will read the string as int sized blocks instead of bytes. This relies on undefined behavior and may fail on some systems, but improves performance on long strings." OFF)
if(LIBC_UNSAFE_STRING_WIDE_READ)
  if(LLVM_USE_SANITIZER)
    message(FATAL_ERROR "LIBC_UNSAFE_STRING_WIDE_READ is set at the same time as a sanitizer. LIBC_UNSAFE_STRING_WIDE_READ causes strlen and memchr to read beyond the end of their target strings, which is undefined behavior caught by sanitizers.")
  else()
    list(APPEND LIBC_COMMON_TUNE_OPTIONS "-DLIBC_UNSAFE_STRING_WIDE_READ")
    endif()
endif()
