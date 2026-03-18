if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
  # If Fortran runtime is built as CUDA library, the linking
  # of targets that link flang_rt.runtime must be done
  # with CUDA_RESOLVE_DEVICE_SYMBOLS.
  # CUDA language must be enabled for CUDA_RESOLVE_DEVICE_SYMBOLS
  # to take effect.
  enable_language(CUDA)
endif()

add_custom_target(FlangUnitTests)
set_target_properties(FlangUnitTests PROPERTIES FOLDER "Flang/Tests")

function(add_flang_unittest test_dirname)
  add_unittest(FlangUnitTests ${test_dirname} ${ARGN})
  if (FLANG_PARALLEL_COMPILE_JOBS)
    set_property(TARGET ${test_dirname} PROPERTY JOB_POOL_COMPILE flang_compile_job_pool)
  endif ()
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_flang_nongtest_unittest test_name)
  cmake_parse_arguments(ARG
    "SLOW_TEST"
    ""
    ""
    ${ARGN})

  if(ARG_SLOW_TEST)
      set(suffix .slow)
  else()
      set(suffix .test)
  endif()

  set(test_filepath "${test_name}.cpp")

  add_executable(${test_name}${suffix} "${test_filepath}")
  set_target_properties(${test_name}${suffix} PROPERTIES FOLDER "Flang/Tests/Unit")
  if (FLANG_PARALLEL_COMPILE_JOBS)
    set_property(TARGET ${test_name}${suffix} PROPERTY JOB_POOL_COMPILE flang_compile_job_pool)
  endif ()

  if (LLVM_LINK_LLVM_DYLIB AND NOT ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
    set(llvm_libs LLVM)
  else()
    llvm_map_components_to_libnames(llvm_libs Support)
  endif()
  target_link_libraries(${test_name}${suffix} ${llvm_libs} ${ARG_UNPARSED_ARGUMENTS})
  set_unittest_link_flags(${test_name}${suffix})

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FlangUnitTests ${test_name}${suffix})
  endif()
endfunction()

add_subdirectory(Optimizer)
add_subdirectory(Common)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Frontend)
