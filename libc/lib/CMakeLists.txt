set(libc_subarchive_entrypoints "")
if(LLVM_LIBC_ENABLE_MINOR_VARIANT)
  if(NOT LLVM_LIBC_SUBARCHIVE_NAME)
    message(FATAL_ERROR
      "LLVM_LIBC_ENABLE_MINOR_VARIANT requires LLVM_LIBC_SUBARCHIVE_NAME to be set.")
  endif()
  if(NOT LLVM_LIBC_SUBARCHIVE_ENTRYPOINTS)
    message(FATAL_ERROR
      "LLVM_LIBC_ENABLE_MINOR_VARIANT requires "
      "LLVM_LIBC_SUBARCHIVE_ENTRYPOINTS to be set.")
  endif()
endif()

set(libc_archive_targets "")
set(libc_archive_names "")
set(libc_archive_entrypoint_lists "")
if(LLVM_LIBC_ENABLE_MINOR_VARIANT)
  foreach(entrypoint IN LISTS LLVM_LIBC_SUBARCHIVE_ENTRYPOINTS)
    if(NOT entrypoint IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
      message(FATAL_ERROR
        "LLVM_LIBC_SUBARCHIVE_ENTRYPOINTS contains '${entrypoint}', which is "
        "not enabled by the target's entrypoints.txt configuration.")
    endif()
    list(APPEND libc_subarchive_entrypoints "${entrypoint}")
  endforeach()
elseif(LLVM_LIBC_FULL_BUILD)
  list(APPEND libc_archive_names c m mvec)
  list(APPEND libc_archive_targets libc libm libmvec)
  list(APPEND libc_archive_entrypoint_lists
    TARGET_LIBC_ENTRYPOINTS TARGET_LIBM_ENTRYPOINTS TARGET_LIBMVEC_ENTRYPOINTS)
else()
  list(APPEND libc_archive_names llvmlibc)
  list(APPEND libc_archive_targets libc)
  list(APPEND libc_archive_entrypoint_lists TARGET_LLVMLIBC_ENTRYPOINTS)
endif()

set(added_archive_targets "")
set(added_shared_targets "")
set(added_bitcode_targets "")
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
    if(TARGET libc-startup)
      add_dependencies(${archive_1} libc-startup)
    endif()
  endif()
  # With LIBC_ENABLE_SHARED the definition of `app` moves out of crt1.o and
  # into libc, so the archive needs it too or a static link is left with the
  # weak declaration resolving to nothing.
  if(LIBC_ENABLE_SHARED AND archive_1 STREQUAL "libc"
    AND TARGET libc.startup.${LIBC_TARGET_OS}.app)
    target_sources(${archive_1} PRIVATE
      $<TARGET_OBJECTS:libc.startup.${LIBC_TARGET_OS}.app>
      $<TARGET_OBJECTS:libc.startup.${LIBC_TARGET_OS}.passive_abi>)
  endif()
  list(APPEND added_archive_targets ${archive_1})

  if(LIBC_ENABLE_SHARED)
    add_entrypoint_library(
      ${archive_1}-shared
      SHARED
      SOVERSION ${LIBC_ABI_VERSION}
      DEPENDS
        ${${archive_2}}
    )
    set_target_properties(
      ${archive_1}-shared
      PROPERTIES
        OUTPUT_NAME ${archive_0}
    )
    target_link_libraries(${archive_1}-shared PUBLIC libc-headers)
    if(TARGET libc-startup)
      add_dependencies(${archive_1}-shared libc-startup)
    endif()
    set(app_object libc.startup.${LIBC_TARGET_OS}.app)
    if(archive_1 STREQUAL "libc")
      if(TARGET ${app_object})
        target_sources(${archive_1}-shared
          PRIVATE $<TARGET_OBJECTS:${app_object}>
          $<TARGET_OBJECTS:libc.startup.${LIBC_TARGET_OS}.passive_abi>)
      endif()
    else()
      # libm and libmvec reference errno and other libc internals.
      target_link_libraries(${archive_1}-shared PRIVATE libc-shared)
    endif()
    list(APPEND added_shared_targets ${archive_1}-shared)
  endif()

  # Add the offloading version of the library for offloading languages. These
  # are installed in the standard search path separate from the other libraries.
  if(LIBC_TARGET_OS_IS_GPU)
    add_bitcode_entrypoint_library(
      ${archive_1}bitcode
      ${archive_1}
      OUTPUT_NAME ${archive_1}.bc
    )
    list(APPEND added_bitcode_targets ${archive_1}bitcode)
  endif()
endforeach()
# Everything is in libc, but a great deal of software still asks for the
# separate libraries it used to be split across, and a link that names one
# fails if there is nothing of that name to find. Empty archives are what
# glibc leaves in their place for the same reason.

if(libc_subarchive_entrypoints)
  add_entrypoint_library(
    ${LLVM_LIBC_SUBARCHIVE_NAME}
    DEPENDS
      ${libc_subarchive_entrypoints}
  )
  set_target_properties(
    ${LLVM_LIBC_SUBARCHIVE_NAME}
    PROPERTIES
      ARCHIVE_OUTPUT_NAME llvm${LLVM_LIBC_SUBARCHIVE_NAME}
  )
  if(LLVM_LIBC_FULL_BUILD)
    target_link_libraries(${LLVM_LIBC_SUBARCHIVE_NAME} PUBLIC libc-headers)
    if(TARGET libc-startup)
      add_dependencies(${LLVM_LIBC_SUBARCHIVE_NAME} libc-startup)
    endif()
  endif()
  list(APPEND added_archive_targets ${LLVM_LIBC_SUBARCHIVE_NAME})
endif()

install(
  TARGETS ${added_archive_targets}
  ARCHIVE DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT libc
)

if(added_shared_targets)
  install(
    TARGETS ${added_shared_targets}
    LIBRARY DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
    COMPONENT libc
  )
endif()

foreach(file ${added_bitcode_targets})
  install(FILES $<TARGET_PROPERTY:${file},TARGET_FILE>
    DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
    COMPONENT libc
  )
endforeach()

set(added_empty_archives "")
if(LLVM_LIBC_FULL_BUILD AND ${LIBC_TARGET_OS} STREQUAL "linux")
  list(APPEND libc_empty_archive_names pthread dl resolv rt util)
  foreach(archive ${libc_empty_archive_names})
    set(empty_archive_path ${LIBC_LIBRARY_DIR}/lib${archive}.a)
    add_custom_command(
      OUTPUT ${empty_archive_path}
      COMMAND ${CMAKE_AR} cqs ${empty_archive_path}
      COMMENT "Creating empty archive ${empty_archive_path}"
      VERBATIM
    )
    install(
      FILES ${empty_archive_path}
      DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
      COMPONENT libc
    )
    add_custom_target(empty-${archive} ALL DEPENDS ${empty_archive_path})
    list(APPEND added_empty_archives empty-${archive})
  endforeach()
  # They are installed with libc, so building libc has to make them.
  add_dependencies(libc ${added_empty_archives})
endif()

set(startup_target "")
if(LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_OS_IS_BAREMETAL)
  # For now we will disable libc-startup installation for baremetal. The
  # correct way to do it would be to make a hookable startup for baremetal
  # and install it as part of the libc installation.
  set(startup_target "libc-startup")
endif()

set(header_install_target "")
if(LLVM_LIBC_FULL_BUILD)
  set(header_install_target install-libc-headers)
endif()

add_custom_target(install-libc
  DEPENDS ${added_archive_targets}
    ${added_shared_targets}
    ${added_bitcode_targets}
    ${added_empty_archives}
    ${startup_target}
    ${header_install_target}
  COMMAND "${CMAKE_COMMAND}"
    -DCMAKE_INSTALL_COMPONENT=libc
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
add_custom_target(install-libc-stripped
  DEPENDS ${added_archive_targets}
    ${added_shared_targets}
    ${added_bitcode_targets}
    ${added_empty_archives}
    ${startup_target}
    ${header_install_target}
  COMMAND "${CMAKE_COMMAND}"
    -DCMAKE_INSTALL_COMPONENT=libc
    -DCMAKE_INSTALL_DO_STRIP=1
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
