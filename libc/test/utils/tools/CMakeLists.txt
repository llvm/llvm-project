add_libc_testsuite(libc-tool-unittests)

function(add_libc_tool_unittest target_name)

  cmake_parse_arguments(
    "LIBC_TOOL_UNITTEST"
    "" # No optional arguments
    "" # Single value arguments
    "SRCS;DEPENDS;ARGS" # Multi-value arguments
    ${ARGN}
  )

  add_executable(${target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_TOOL_UNITTEST_SRCS}
  )
  target_link_libraries(${target_name}
    PRIVATE
    llvm_gtest_main
    llvm_gtest
    ${LIBC_TOOL_UNITTEST_DEPENDS}
  )

  add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND $<TARGET_FILE:${target_name}>
            ${LIBC_TOOL_UNITTEST_ARGS}
  )
  add_dependencies(libc-tool-unittests ${target_name})

  target_compile_options(${target_name} PUBLIC -fno-rtti)
  target_link_libraries(${target_name} PRIVATE LLVMSupport)
endfunction()

# Build the gtest library needed for unittests if we do not have it already.
if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest AND NOT TARGET llvm_gtest)
  include_directories(${LLVM_LIBC_INCLUDE_DIRS})
  add_subdirectory(${LLVM_THIRD_PARTY_DIR}/unittest third-party/unittest)
endif()

add_subdirectory(WrapperGen)
