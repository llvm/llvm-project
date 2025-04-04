include(CMakePushCheckState)
include(CheckCompilerFlag)

function(llvm_check_compiler_linker_flag lang reset target_type flag out_var)
  # If testing a flag with check_compiler_flag, it gets added to the compile
  # command only, but not to the linker command in that test. If the flag
  # is vital for linking to succeed, the test would fail even if it would
  # have succeeded if it was included on both commands.
  #
  # Therefore, try adding the flag to CMAKE_REQUIRED_FLAGS, which gets
  # added to both compiling and linking commands in the tests.

  # In some cases, we need to disable existing linker options completely;
  # e.g. due to https://gitlab.kitware.com/cmake/cmake/-/issues/23454, in the
  # context of CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG, for example. To this end,
  # we the RESET option of cmake_push_check_state, c.f.
  # https://cmake.org/cmake/help/latest/module/CMakePushCheckState.html
  #
  # Due to the same CMake issue, we need to be able to override the targe type,
  # as some checks will fail by default for shared libraries. A concrete example
  # is checking for `-funwind-tables` when building libunwind (e.g. for ARM EHABI).
  #
  # This happens because, when performing CMake checks, adding `-funwind-tables`
  # for a dynamic target causes the check to produce a false negative, because the
  # compiler compiler generates calls to `__aeabi_unwind_cpp_pr0`, which is defined
  # in libunwind itself, which isn't built yet, so the linker complains about
  # undefined symbols. This would lead to libunwind not being built with this flag,
  # which makes libunwind quite useless in this setup.
  cmake_push_check_state(${reset})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flag}")
  if (NOT target_type STREQUAL "")
    set(_previous_CMAKE_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  endif()
  # run the actual check
  check_compiler_flag("${lang}" "" ${out_var})
  if (NOT target_type STREQUAL "")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_previous_CMAKE_TRY_COMPILE_TARGET_TYPE})
  endif()
  cmake_pop_check_state()
endfunction()
