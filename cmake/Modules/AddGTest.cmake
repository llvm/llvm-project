
function (build_gtest gtest_name)
  cmake_parse_arguments(ARG "LLVM_SUPPORT" "" "" ${ARGN})

  if (ARG_LLVM_SUPPORT)
    set(GTEST_USE_LLVM 1)
  else ()
    set(GTEST_USE_LLVM 0)
  endif ()
  add_subdirectory("${LLVM_THIRD_PARTY_DIR}/unittest" "${CMAKE_BINARY_DIR}/third-party/${gtest_name}_gtest")
endfunction ()
