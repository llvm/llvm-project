if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_object_library(
    sigsetjmp_epilogue
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.sigsetjmp_epilogue
  )
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

if (TARGET libc.src.setjmp.sigsetjmp_epilogue)
  add_entrypoint_object(
    siglongjmp
    SRCS
      siglongjmp.cpp
    HDRS
      siglongjmp.h
    DEPENDS
      .longjmp
  )

  add_entrypoint_object(
    sigsetjmp
    ALIAS
    DEPENDS
      .${LIBC_TARGET_ARCHITECTURE}.sigsetjmp
  )
endif()
