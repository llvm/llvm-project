if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()
add_custom_target(stdlib-integration-tests)
add_dependencies(libc-integration-tests stdlib-integration-tests)

add_integration_test(
  getenv_test
  SUITE
    stdlib-integration-tests
  SRCS
    getenv_test.cpp
  DEPENDS
    libc.src.stdlib.getenv
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

# Environment variable functions only make sense in full build mode.
if(LLVM_LIBC_FULL_BUILD)
  add_integration_test(
    setenv_test
    SUITE
      stdlib-integration-tests
    SRCS
      setenv_test.cpp
    DEPENDS
      libc.src.stdlib.getenv
      libc.src.stdlib.linux.setenv
      libc.src.string.strcmp
  )
endif()
