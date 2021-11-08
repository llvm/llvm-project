include(CMakePushCheckState)
include(CheckCCompilerFlag)

function(llvm_check_linker_flag flag dest)
  # If testing a flag with check_c_compiler_flag, it gets added to the compile
  # command only, but not to the linker command in that test. If the flag
  # is vital for linking to succeed, the test would fail even if it would
  # have succeeded if it was included on both commands.
  #
  # Therefore, try adding the flag to CMAKE_REQUIRED_FLAGS, which gets
  # added to both compiling and linking commands in the tests.

  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flag}")
  check_c_compiler_flag("" ${dest})
  cmake_pop_check_state()
endfunction()
