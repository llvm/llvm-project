add_subdirectory(assert)
add_subdirectory(ctype)
add_subdirectory(errno)
add_subdirectory(fenv)
add_subdirectory(math)
add_subdirectory(signal)
add_subdirectory(stdio)
add_subdirectory(stdlib)
add_subdirectory(string)
add_subdirectory(sys)
add_subdirectory(threads)
add_subdirectory(time)
add_subdirectory(unistd)

set(public_test ${CMAKE_CURRENT_BINARY_DIR}/public_integration_test.cpp)

set(entrypoints_name_list "")
foreach(entry IN LISTS TARGET_LLVMLIBC_ENTRYPOINTS)
  get_target_property(entry_name ${entry} "ENTRYPOINT_NAME")
  list(APPEND entrypoints_name_list ${entry_name})
endforeach()

# TODO: Remove these when they are added to the TableGen.
list(REMOVE_ITEM entrypoints_name_list "__assert_fail" "__errno_location")
list(TRANSFORM entrypoints_name_list PREPEND "-e=")

file(GLOB spec_files ${LIBC_SOURCE_DIR}/spec/*.td)

# Generate integration test souce code.
add_custom_command(
  OUTPUT ${public_test}
  COMMAND $<TARGET_FILE:libc-prototype-testgen> -o ${public_test}
          ${entrypoints_name_list}
          -I ${LIBC_SOURCE_DIR}
          ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td

  DEPENDS ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td ${spec_files}
          libc-prototype-testgen ${TARGET_PUBLIC_HEADERS}
          llvmlibc
)

add_executable(
  libc-integration-test
  EXCLUDE_FROM_ALL
  ${public_test}
)
# Blank out default include directories to prevent accidentally including
# system headers or our own internal headers.
set_target_properties(
  libc-integration-test
  PROPERTIES
  INCLUDE_DIRECTORIES ""
)
# Only include we need is the include for cpp::IsSame and our generated
# public headers.
target_include_directories(
  libc-integration-test BEFORE
  PRIVATE
    "${LIBC_SOURCE_DIR}/utils/CPP"
    "${LIBC_BUILD_DIR}/include"
)
target_compile_options(
  libc-integration-test
  PRIVATE
  -ffreestanding
)
target_link_options(
  libc-integration-test
  PRIVATE "-nostdlib"
)
set(library_files)
foreach(library_name IN LISTS "llvmlibc")
  get_target_property(library_file ${library_name} "LIBRARY_FILE")
  list(APPEND library_files ${library_file})
endforeach()

if(COMPILER_RESOURCE_DIR AND LLVM_LIBC_ENABLE_LINTING)
  add_custom_target(
    libc-integration-test-tidy
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
  add_dependencies(libc-integration-test libc-integration-test-tidy)
endif()

target_link_libraries(libc-integration-test
  PRIVATE
  ${library_files}
)
