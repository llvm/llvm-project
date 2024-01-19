set(libc_archive_targets "")
set(libc_archive_names "")
set(libc_archive_entrypoint_lists "")
if(LLVM_LIBC_FULL_BUILD)
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    list(APPEND libc_archive_names cgpu mgpu)
  else()
    list(APPEND libc_archive_names c m)
  endif()
  list(APPEND libc_archive_targets libc libm)
  list(APPEND libc_archive_entrypoint_lists
       TARGET_LIBC_ENTRYPOINTS TARGET_LIBM_ENTRYPOINTS)
else()
  list(APPEND libc_archive_names llvmlibc)
  list(APPEND libc_archive_targets libc)
  list(APPEND libc_archive_entrypoint_lists TARGET_LLVMLIBC_ENTRYPOINTS)
endif()

set(added_archive_targets "")
foreach(archive IN ZIP_LISTS
        libc_archive_names libc_archive_targets libc_archive_entrypoint_lists)
  if(NOT ${archive_2})
    # If an entrypoint list is missing, then skip adding targets for it.
    continue()
  endif()
  add_entrypoint_library(
    ${archive_1}
    DEPENDS
      ${${archive_2}}
  )
  set_target_properties(
    ${archive_1}
    PROPERTIES
      ARCHIVE_OUTPUT_NAME ${archive_0}
  )
  if(LLVM_LIBC_FULL_BUILD)
    target_link_libraries(${archive_1} PUBLIC libc-headers)
  endif()
  list(APPEND added_archive_targets ${archive_1})
endforeach()

if(LIBC_TARGET_TRIPLE)
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LIBC_TARGET_TRIPLE})
elseif(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT LIBC_GPU_BUILD)
  set(LIBC_INSTALL_LIBRARY_DIR
      lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX})
endif()

install(
  TARGETS ${added_archive_targets}
  ARCHIVE DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT libc
)

if(NOT LIBC_TARGET_OS_IS_BAREMETAL)
  # For now we will disable libc-startup installation for baremetal. The
  # correct way to do it would be to make a hookable startup for baremetal
  # and install it as part of the libc installation.
  set(startup_target "libc-startup")
endif()

if(LLVM_LIBC_FULL_BUILD)
  set(header_install_target install-libc-headers)
endif()

add_custom_target(install-libc
                  DEPENDS ${added_archive_targets}
                          ${startup_target}
                          ${header_install_target}
                  COMMAND "${CMAKE_COMMAND}"
                          -DCMAKE_INSTALL_COMPONENT=libc
                          -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
add_custom_target(install-libc-stripped
                  DEPENDS ${added_archive_targets}
                          ${startup_target}
                          ${header_install_target}
                  COMMAND "${CMAKE_COMMAND}"
                          -DCMAKE_INSTALL_COMPONENT=libc
                          -DCMAKE_INSTALL_DO_STRIP=1
                          -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
