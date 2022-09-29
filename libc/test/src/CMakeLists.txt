function(add_fp_unittest name)
  cmake_parse_arguments(
    "MATH_UNITTEST"
    "NEED_MPFR" # Optional arguments
    "" # Single value arguments
    "LINK_LIBRARIES" # Multi-value arguments
    ${ARGN}
  )

  if(MATH_UNITTEST_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message("WARNING: Math test ${name} will be skipped as MPFR library is not available.")
      return()
    endif()
  endif()

  if(MATH_UNITTEST_NEED_MPFR)
    list(APPEND MATH_UNITTEST_LINK_LIBRARIES libcMPFRWrapper -lmpfr -lgmp)
  endif()
  list(APPEND MATH_UNITTEST_LINK_LIBRARIES LibcFPTestHelpers)

  add_libc_unittest(
    ${name}
    LINK_LIBRARIES "${MATH_UNITTEST_LINK_LIBRARIES}"
    "${MATH_UNITTEST_UNPARSED_ARGUMENTS}"
  )
endfunction(add_fp_unittest)

add_subdirectory(__support)
add_subdirectory(ctype)
add_subdirectory(errno)
add_subdirectory(fenv)
add_subdirectory(inttypes)
add_subdirectory(math)
add_subdirectory(string)
add_subdirectory(stdlib)
add_subdirectory(stdio)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_subdirectory(fcntl)
  add_subdirectory(sched)
  add_subdirectory(sys)
  add_subdirectory(unistd)
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(dirent)
# The signal API is currently disabled as signal.h is incorrect.
# since assert uses the signal API, we disable assert also.
# add_subdirectory(assert)
# add_subdirectory(signal)
add_subdirectory(time)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_subdirectory(pthread)
endif()

set(public_test ${CMAKE_CURRENT_BINARY_DIR}/public_api_test.cpp)

set(entrypoints_name_list "")
foreach(entry IN LISTS TARGET_LLVMLIBC_ENTRYPOINTS)
  get_target_property(entry_name ${entry} "ENTRYPOINT_NAME")
  list(APPEND entrypoints_name_list ${entry_name})
endforeach()

# TODO: Remove these when they are added to the TableGen.
list(REMOVE_ITEM entrypoints_name_list "__assert_fail" "__errno_location")
list(TRANSFORM entrypoints_name_list PREPEND "-e=")

file(GLOB spec_files ${LIBC_SOURCE_DIR}/spec/*.td)

# Generate api test souce code.
add_custom_command(
  OUTPUT ${public_test}
  COMMAND $<TARGET_FILE:libc-prototype-testgen> -o ${public_test}
          ${entrypoints_name_list}
          -I ${LIBC_SOURCE_DIR}
          ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td

  DEPENDS ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td ${spec_files}
          libc-prototype-testgen ${TARGET_PUBLIC_HEADERS}
          ${LIBC_TARGET}
)

add_executable(
  libc-api-test
  EXCLUDE_FROM_ALL
  ${public_test}
)
# Blank out default include directories to prevent accidentally including
# system headers or our own internal headers.
set_target_properties(
  libc-api-test
  PROPERTIES
  INCLUDE_DIRECTORIES ""
)
target_link_libraries(libc-api-test ${LIBC_TARGET})

# Only include we need is the include for cpp::IsSame and our generated
# public headers.
target_include_directories(
  libc-api-test BEFORE
  PRIVATE
    "${LIBC_SOURCE_DIR}/src/__support/CPP"
    "${LIBC_BUILD_DIR}/include"
)
target_compile_options(
  libc-api-test
  PRIVATE
  -ffreestanding
)
target_link_options(
  libc-api-test
  PRIVATE "-nostdlib"
)
set(library_files)
foreach(library_name IN LISTS "llvmlibc")
  get_target_property(library_file ${library_name} "LIBRARY_FILE")
  list(APPEND library_files ${library_file})
endforeach()

if(COMPILER_RESOURCE_DIR AND LLVM_LIBC_ENABLE_LINTING)
  add_custom_target(
    libc-api-test-tidy
    VERBATIM
    COMMAND $<TARGET_FILE:clang-tidy> --system-headers
      --checks=-*,llvmlibc-restrict-system-libc-headers
      "--extra-arg=-resource-dir=${COMPILER_RESOURCE_DIR}"
      --header-filter=.*
      --warnings-as-errors=llvmlibc-*
      "-config={CheckOptions: [{key: llvmlibc-restrict-system-libc-headers.Includes, value: '-*, linux/*, asm/*.h, asm-generic/*.h'}]}"
      --quiet
      -p ${PROJECT_BINARY_DIR}
      ${public_test}
    DEPENDS
      clang-tidy ${public_test}
  )
  add_dependencies(libc-api-test libc-api-test-tidy)
endif()

target_link_libraries(libc-api-test
  PRIVATE
  ${library_files}
)
