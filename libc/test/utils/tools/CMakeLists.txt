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
    gtest_main
    gtest
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

add_subdirectory(WrapperGen)
