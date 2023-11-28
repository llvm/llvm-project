add_libc_exhaustive_testsuite(libc-math-gpu-exhaustive-tests)

if((NOT "openmp" IN_LIST LLVM_ENABLE_RUNTIMES) AND
   (NOT "openmp" IN_LIST LLVM_ENABLE_PROJECTS))
  message(STATUS "The 'openmp' runtime must be enabled to run exhaustive "
                 "GPU tests.")
  return()
endif()

# Attempt to locate the libraries required for offloading.
if(TARGET omptarget.devicertl AND TARGET omptarget AND TARGET omp)
  set(LIBC_OPENMP_RUNTIME omptarget.devicertl omptarget omp)
else()
  find_library(omptarget.devicertl NAMES omptarget.devicertl
               PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR} ${LLVM_LIBRARY_DIR}
               NO_DEFAULT_PATH
  )
  find_library(omptarget NAMES omptarget
               PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR} ${LLVM_LIBRARY_DIR}
               NO_DEFAULT_PATH
  )
  find_library(omp NAMES omp
               PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR} ${LLVM_LIBRARY_DIR}
               NO_DEFAULT_PATH
  )
  if(NOT omptarget.devicertl OR NOT omptarget OR NOT omp)
    message(WARNING "Could not find the OpenMP runtime for exhaustive tests")
    return()
  endif()
  set(LIBC_OPENMP_RUNTIME ${omptarget.devicertl} ${omptarget} ${omp})
endif()

# Ensure that the tests do not use any other libraries found on the system.
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND LIBC_OPENMP_RUNTIME "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
                                  "-Wl,-rpath,${LLVM_LIBRARY_OUTPUT_INTDIR}")
endif()

function(add_gpu_exhaustive_test name)
  cmake_parse_arguments(
    "MATH_GPU_EXHAUSTIVE_TEST"
    "NEED_MPFR" # Optional arguments
    "" # Single value arguments
    "SRCS;HDRS;DEPENDS;ARGS;ENV;COMPILE_OPTIONS;LINK_LIBRARIES" # Multi-value arguments
    ${ARGN}
  )

  if(NOT MATH_GPU_EXHAUSTIVE_TEST_SRCS)
    message(FATAL_ERROR "'add_gpu_exhaustive_test' target requires a SRCS list "
                        "of .cpp files.")
  endif()
  if(NOT MATH_GPU_EXHAUSTIVE_TEST_DEPENDS)
    message(FATAL_ERROR "'add_gpu_exhaustive_test' target requires a DEPENDS "
                        "list of 'add_entrypoint_object' targets.")
  endif()

  if(MATH_GPU_EXHAUSTIVE_TEST_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message(VERBOSE "Math test ${name} will be skipped as MPFR library is "
                      "not available.")
      return()
    endif()
  endif()


  if(MATH_GPU_EXHAUSTIVE_TEST_NEED_MPFR)
    list(APPEND MATH_GPU_EXHAUSTIVE_TEST_LINK_LIBRARIES libcMPFRWrapper)
  endif()

  get_fq_target_name(${name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${MATH_GPU_EXHAUSTIVE_TEST_DEPENDS})
  list(REMOVE_DUPLICATES fq_deps_list)

  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    message(STATUS "Skipping unittest ${fq_target_name} as it has missing deps:"
                   " ${skipped_entrypoints_list}.")
    return()
  endif()

  list(APPEND MATH_GPU_EXHAUSTIVE_TEST_LINK_LIBRARIES 
              ${LIBC_OPENMP_RUNTIME} "--offload-link")
  foreach(link_object_file ${link_object_files})
    list(APPEND MATH_GPU_EXHAUSTIVE_TEST_LINK_LIBRARIES 
                "-Xoffload-linker ${link_object_file}")
  endforeach()

  set(fq_exhaustive_target_name ${fq_target_name}.__exhaustive__)
  add_executable(
    ${fq_exhaustive_target_name}
    EXCLUDE_FROM_ALL
    ${MATH_GPU_EXHAUSTIVE_TEST_SRCS}
    ${MATH_GPU_EXHAUSTIVE_TEST_HDRS}
  )
  set(LIBC_GPU_TEST_OPTIONS -fopenmp -nogpulib -nogpuinc -foffload-lto -fno-rtti
                            -fopenmp-offload-mandatory -fpie -fno-exceptions
                            --offload-arch=${LIBC_GPU_TARGET_ARCHITECTURE})

  target_include_directories(${fq_exhaustive_target_name} PRIVATE 
                             ${LIBC_SOURCE_DIR})

  target_compile_options(${fq_exhaustive_target_name} PRIVATE 
                         ${MATH_GPU_EXHAUSTIVE_TEST_COMPILE_OPTIONS} 
                         ${LIBC_GPU_TEST_OPTIONS}
  )
  target_link_libraries(${fq_exhaustive_target_name} PRIVATE 
                        ${MATH_GPU_EXHAUSTIVE_TEST_LINK_LIBRARIES} LibcTest.unit
  )

  add_dependencies(${fq_exhaustive_target_name} ${fq_deps_list})
  set_target_properties(${fq_exhaustive_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_custom_target(
    ${fq_target_name}
    COMMAND OMP_TARGET_OFFLOAD=MANDATORY 
            $<TARGET_FILE:${fq_exhaustive_target_name}>
    COMMAND_EXPAND_LISTS
    COMMENT "Running exhaustive GPU test ${fq_target_name}"
  )
  add_dependencies(libc-math-gpu-exhaustive-tests ${fq_target_name})
endfunction()

add_gpu_exhaustive_test(
  truncf_test
  NEED_MPFR
  SRCS
    truncf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.truncf
    libc.src.__support.FPUtil.fp_bits
)
