add_subdirectory(timing)

add_custom_target(gpu-benchmark)

function (add_gpu_benchmark test_name)
  if(NOT TARGET libc.startup.${LIBC_TARGET_OS}.crt1)
    message(VERBOSE "Skipping ${fq_target_name} as it is not available on ${LIBC_TARGET_OS}.")
    return()
  endif()

  cmake_parse_arguments(
    "GPU_BENCHMARK"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;HDRS;DEPENDS;ARGS;ENV;COMPILE_OPTIONS;LINK_LIBRARIES;LOADER_ARGS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT GPU_BENCHMARK_SUITE)
    message(FATAL_ERROR "SUITE not specified for ${fq_target_name}")
  endif()
  if(NOT GPU_BENCHMARK_SRCS)
    message(FATAL_ERROR "The SRCS list for add_gpu_benchmark is missing.")
  endif()

  get_fq_target_name(${test_name} fq_target_name)
  get_fq_target_name(${test_name}.libc fq_libc_target_name) # Stores the compiled libc + infrastructure archive to link in
  get_fq_deps_list(fq_deps_list ${GPU_BENCHMARK_DEPENDS})
  list(APPEND fq_deps_list
      # Hermetic tests use the platform's startup object. So, their deps also
      # have to be collected.
      libc.startup.${LIBC_TARGET_OS}.crt1
      # We always add the memory functions objects. This is because the
      # compiler's codegen can emit calls to the C memory functions.
      libc.src.string.bcmp
      libc.src.string.bzero
      libc.src.string.memcmp
      libc.src.string.memcpy
      libc.src.string.memmove
      libc.src.string.memset
      libc.src.__support.StringUtil.error_to_string
  )

  list(REMOVE_DUPLICATES fq_deps_list)

  # TODO: Instead of gathering internal object files from entrypoints,
  # collect the object files with public names of entrypoints.
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Skipping hermetic test ${fq_target_name} as it has missing deps: "
              "${skipped_entrypoints_list}.")
    endif()
    return()
  endif()
  list(REMOVE_DUPLICATES link_object_files)

  # Make a library of all deps
  add_library(
    ${fq_target_name}.__libc__
    STATIC
    EXCLUDE_FROM_ALL
    ${link_object_files}
  )
  set_target_properties(${fq_target_name}.__libc__
      PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_target_properties(${fq_target_name}.__libc__
      PROPERTIES ARCHIVE_OUTPUT_NAME ${fq_target_name}.libc)

  set(fq_build_target_name ${fq_target_name}.__build__)
  add_executable(
    ${fq_build_target_name}
    EXCLUDE_FROM_ALL
    $<$<BOOL:${LIBC_TARGET_ARCHITECTURE_IS_NVPTX}>:${link_object_files}>
    ${GPU_BENCHMARK_SRCS}
    ${GPU_BENCHMARK_HDRS}
  )
  set_target_properties(${fq_build_target_name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  _get_hermetic_test_compile_options(compile_options "${GPU_BENCHMARK_COMPILE_OPTIONS}")
  target_include_directories(${fq_build_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  _get_hermetic_test_compile_options(compile_options "${GPU_BENCHMARK_COMPILE_OPTIONS}")
  target_compile_options(${fq_build_target_name} PRIVATE ${compile_options}) 

  set(link_libraries "")
  foreach(lib in LISTS GPU_BENCHMARK_LINK_LIBRARIES)
    if(TARGET ${lib}.hermetic)
      list(APPEND link_libraries ${lib}.hermetic)
    else()
      list(APPEND link_libraries ${lib})
    endif()
  endforeach()

  if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
    target_link_options(${fq_build_target_name} PRIVATE
      ${LIBC_COMPILE_OPTIONS_DEFAULT}
      -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE} -flto -Wno-multi-gpu
      "-Wl,-mllvm,-amdgpu-lower-global-ctor-dtor=0" -nostdlib -static
      "-Wl,-mllvm,-amdhsa-code-object-version=${LIBC_GPU_CODE_OBJECT_VERSION}")
  elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
    # We need to use the internal object versions for NVPTX.
    set(internal_suffix ".__internal__")
    target_link_options(${fq_build_target_name} PRIVATE
      ${LIBC_COMPILE_OPTIONS_DEFAULT} -Wno-multi-gpu
      "-Wl,--suppress-stack-size-warning"
      -march=${LIBC_GPU_TARGET_ARCHITECTURE} -nostdlib -static
      "--cuda-path=${LIBC_CUDA_ROOT}")
  elseif(LIBC_CC_SUPPORTS_NOSTDLIBPP)
    target_link_options(${fq_build_target_name} PRIVATE -nolibc -nostartfiles -nostdlib++ -static)
  else()
    # Older version of gcc does not support `nostdlib++` flag.  We use
    # `nostdlib` and link against libgcc_s, which cannot be linked statically.
    target_link_options(${fq_build_target_name} PRIVATE -nolibc -nostartfiles -nostdlib)
    list(APPEND link_libraries ${LIBGCC_S_LOCATION})
  endif()

  # link libraries for the BUILD target (i.e. to compile the test)
  target_link_libraries(
    ${fq_build_target_name}
    PRIVATE
      libc.startup.${LIBC_TARGET_OS}.crt1${internal_suffix}
      ${link_libraries}
      # LibcTest.hermetic
      LibcGpuBenchmark.hermetic
      # LibcHermeticTestSupport.hermetic
      LibcHermeticTestSupport.hermetic
      # The NVIDIA 'nvlink' linker does not currently support static libraries.
      $<$<NOT:$<BOOL:${LIBC_TARGET_ARCHITECTURE_IS_NVPTX}>>:${fq_target_name}.__libc__>)

  add_dependencies(${fq_build_target_name}
    LibcGpuBenchmark.hermetic
  ${fq_deps_list})

  # Tests on the GPU require an external loader utility to launch the kernel.
  if(TARGET libc.utils.gpu.loader)
    add_dependencies(${fq_build_target_name} libc.utils.gpu.loader)
    get_target_property(gpu_loader_exe libc.utils.gpu.loader "EXECUTABLE")
  endif()

  set(test_cmd ${GPU_BENCHMARK_ENV}
      $<$<BOOL:${LIBC_TARGET_OS_IS_GPU}>:${gpu_loader_exe}> ${CMAKE_CROSSCOMPILING_EMULATOR} ${GPU_BENCHMARK_LOADER_ARGS}
      $<TARGET_FILE:${fq_build_target_name}> ${GPU_BENCHMARK_ARGS})
  add_custom_target(
    ${fq_target_name}
    COMMAND ${test_cmd}
    COMMAND_EXPAND_LISTS
    COMMENT "Running GPU benchmark ${fq_target_name}"
  )

  # Make this benchmark part of its suite
  add_dependencies(${GPU_BENCHMARK_SUITE} ${fq_target_name})
  # Remember to make this benchmark part of the umbrella command
  add_dependencies(gpu-benchmark ${fq_target_name})
endfunction(add_gpu_benchmark)

add_unittest_framework_library(
  LibcGpuBenchmark
  SRCS
    LibcGpuBenchmark.cpp
    LibcGpuBenchmarkMain.cpp
    BenchmarkLogger.cpp
  HDRS
    LibcGpuBenchmark.h
    BenchmarkLogger.h
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.macros.properties.types
    libc.src.__support.OSUtil.osutil
    libc.src.__support.uint128
    libc.benchmarks.gpu.timing.timing
)

add_subdirectory(src)
