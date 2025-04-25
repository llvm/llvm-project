if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
  # If Fortran runtime is built as CUDA library, the linking
  # of targets that link FortranRuntime must be done
  # with CUDA_RESOLVE_DEVICE_SYMBOLS.
  # CUDA language must be enabled for CUDA_RESOLVE_DEVICE_SYMBOLS
  # to take effect.
  enable_language(CUDA)
endif()

add_custom_target(FlangRTUnitTests)
set_target_properties(FlangRTUnitTests PROPERTIES FOLDER "Flang-rt Unit Tests")

function(add_flang_rt_unittest_offload_properties target)
  # Set CUDA_RESOLVE_DEVICE_SYMBOLS.
  if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
    set_target_properties(${target}
      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
  endif()
  # Enable OpenMP offload during linking. We may need to replace
  # LINK_OPTIONS with COMPILE_OPTIONS when there are OpenMP offload
  # unittests.
  #
  # FIXME: replace 'native' in --offload-arch option with the list
  #        of targets that Fortran Runtime was built for.
  #        Common code must be moved from flang/runtime/CMakeLists.txt.
  # TODO: Revisit this because of Flang-rt. runtime is no longer an added subdirectory of flang.
  if (NOT FLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD STREQUAL "off")
    set_target_properties(${target}
        PROPERTIES LINK_OPTIONS
        "-fopenmp;--offload-arch=native"
    )
  endif()
endfunction()

if(NOT TARGET llvm_gtest)
#  need to install ?
#  message(FATAL_ERROR "Target llvm_gtest not found.")
endif()

function(add_flang_rt_unittest test_dirname)
  add_unittest(FlangRTUnitTests ${test_dirname} ${ARGN})
  add_flang_rt_unittest_offload_properties(${test_dirname})
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_flang_rt_nongtest_unittest test_name)
  cmake_parse_arguments(ARG
    "SLOW_TEST"
    ""
    ""
    ${ARGN})

  list(APPEND LLVM_COMPILE_FLAGS "-L${LLVM_BINARY_DIR}/lib")
  if(ARG_SLOW_TEST)
      set(suffix .slow)
  else()
      set(suffix .test)
  endif()

  add_executable(${test_name}${suffix} ${test_name}.cpp)

  if (LLVM_LINK_LLVM_DYLIB AND NOT ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
    set(llvm_libs LLVM)
  else()
    llvm_map_components_to_libnames(llvm_libs Support)
  endif()
  target_link_libraries(${test_name}${suffix} ${llvm_libs} ${ARG_UNPARSED_ARGUMENTS})

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FlangRTUnitTests ${test_name}${suffix})
  endif()

  add_flang_rt_unittest_offload_properties(${test_name}${suffix})
endfunction()

add_subdirectory(FortranRuntime)
# TODO: We may want to find a better location for these tests that use the runtime
add_subdirectory(FortranEvaluate)
