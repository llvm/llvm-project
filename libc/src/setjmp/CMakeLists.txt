if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

if (LIBC_CONF_SETJMP_ENABLE_FORTIFICATION)
  if (TARGET libc.src.setjmp.${LIBC_TARGET_OS}.checksum
      AND LIBC_TARGET_ARCHITECTURE STREQUAL "x86_64")
    add_object_library(
      checksum
      ALIAS
      DEPENDS
        .${LIBC_TARGET_OS}.checksum
    )
    set(fortification_deps libc.src.setjmp.checksum)
    set(fortification_defs -DLIBC_COPT_SETJMP_ENABLE_FORTIFICATION=1)
  else()
    message(WARNING "Jmpbuf fortification is enabled but not supported for target ${LIBC_TARGET_ARCHITECTURE} ${LIBC_TARGET_OS}")
    set(fortification_deps)
    set(fortification_defs -DLIBC_COPT_SETJMP_ENABLE_FORTIFICATION=0)
  endif()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
endif()

add_entrypoint_object(
  setjmp
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.setjmp
)

add_entrypoint_object(
  longjmp
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.longjmp
)
