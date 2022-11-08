function(add_example name)
  add_executable(
    ${name}
    ${ARGN}
  )

  if(LIBC_FULLBUILD)
    target_link_options(${name} PRIVATE -static -rtlib=compiler-rt -fuse-ld=lld)
  elseif(LIBC_OVERLAY_ARCHIVE_DIR)
    target_link_directories(${name} PRIVATE ${LIBC_OVERLAY_ARCHIVE_DIR})
    target_link_options(${name} PRIVATE -l:libllvmlibc.a)
  else()
    message(FATAL_ERROR "Either LIBC_FULLBUILD should be on or "
                        "LIBC_OVERLAY_ARCHIVE_DIR should be set.")
  endif()
endfunction()
