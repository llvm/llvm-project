# This is a helper function and not a build rule. It is to be used by the
# various test rules to generate the full list of object files
# recursively produced by "add_entrypoint_object" and "add_object_library"
# targets.
# Usage:
#   get_object_files_for_test(<result var>
#                             <skipped_entrypoints_var>
#                             <target0> [<target1> ...])
#
#   The list of object files is collected in <result_var>.
#   If skipped entrypoints were found, then <skipped_entrypoints_var> is
#   set to a true value.
#   targetN is either an "add_entrypoint_target" target or an
#   "add_object_library" target.
function(get_object_files_for_test result skipped_entrypoints_list)
  set(object_files "")
  set(skipped_list "")
  foreach(dep IN LISTS ARGN)
    if (NOT TARGET ${dep})
      # Skip any tests whose dependencies have not been defined.
      list(APPEND skipped_list ${dep})
      continue()
    endif()
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT dep_type)
      # Target for which TARGET_TYPE property is not set do not
      # provide any object files.
      continue()
    endif()

    if(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      get_target_property(dep_object_files ${dep} "OBJECT_FILES")
      if(dep_object_files)
        list(APPEND object_files ${dep_object_files})
      endif()
    elseif(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      get_target_property(is_skipped ${dep} "SKIPPED")
      if(is_skipped)
        list(APPEND skipped_list ${dep})
        continue()
      endif()
      get_target_property(object_file_raw ${dep} "OBJECT_FILE_RAW")
      if(object_file_raw)
        list(APPEND object_files ${object_file_raw})
      endif()
    elseif(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})
      # We skip tests for all externally implemented entrypoints.
      list(APPEND skipped_list ${dep})
      continue()
    endif()

    get_target_property(indirect_deps ${dep} "DEPS")
    get_object_files_for_test(
        indirect_objfiles indirect_skipped_list ${indirect_deps})
    list(APPEND object_files ${indirect_objfiles})
    if(indirect_skipped_list)
      list(APPEND skipped_list ${indirect_skipped_list})
    endif()
  endforeach(dep)
  list(REMOVE_DUPLICATES object_files)
  set(${result} ${object_files} PARENT_SCOPE)
  list(REMOVE_DUPLICATES skipped_list)
  set(${skipped_entrypoints_list} ${skipped_list} PARENT_SCOPE)
endfunction(get_object_files_for_test)

# Rule to add a libc unittest.
# Usage
#    add_libc_unittest(
#      <target name>
#      SUITE <name of the suite this test belongs to>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#      COMPILE_OPTIONS <list of special compile options for this target>
#      LINK_LIBRARIES <list of linking libraries for this target>
#    )
function(create_libc_unittest fq_target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    "LIBC_UNITTEST"
    "NO_RUN_POSTBUILD" # Optional arguments
    "SUITE;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;LINK_LIBRARIES;FLAGS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_UNITTEST_SRCS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_UNITTEST_DEPENDS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_deps_list(fq_deps_list ${LIBC_UNITTEST_DEPENDS})
  list(APPEND fq_deps_list libc.src.__support.StringUtil.error_to_string
                           libc.test.UnitTest.ErrnoSetterMatcher)
  list(REMOVE_DUPLICATES fq_deps_list)
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    # If a test is OS/target machine independent, it has to be skipped if the
    # OS/target machine combination does not provide any dependent entrypoints.
    # If a test is OS/target machine specific, then such a test will live is a
    # OS/target machine specific directory and will be skipped at the directory
    # level if required.
    #
    # There can potentially be a setup like this: A unittest is setup for a
    # OS/target machine independent object library, which in turn depends on a
    # machine specific object library. Such a test would be testing internals of
    # the libc and it is assumed that they will be rare in practice. So, they
    # can be skipped in the corresponding CMake files using platform specific
    # logic. This pattern is followed in the startup tests for example.
    #
    # Another pattern that is present currently is to detect machine
    # capabilities and add entrypoints and tests accordingly. That approach is
    # much lower level approach and is independent of the kind of skipping that
    # is happening here at the entrypoint level.
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Skipping unittest ${fq_target_name} as it has missing deps: "
              "${skipped_entrypoints_list}.")
      message(STATUS ${msg})
    endif()
    return()
  endif()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding unit test ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  if(LIBC_UNITTEST_NO_RUN_POSTBUILD)
    set(fq_build_target_name ${fq_target_name})
  else()
    set(fq_build_target_name ${fq_target_name}.__build__)
  endif()

  add_executable(
    ${fq_build_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_UNITTEST_SRCS}
    ${LIBC_UNITTEST_HDRS}
  )
  target_include_directories(${fq_build_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  target_compile_options(
    ${fq_build_target_name}
    PRIVATE -fpie ${LIBC_COMPILE_OPTIONS_DEFAULT}
  )
  if(LLVM_LIBC_FULL_BUILD)
    target_compile_options(
      ${fq_build_target_name}
      PRIVATE -ffreestanding
    )
  endif()
  if(LIBC_UNITTEST_COMPILE_OPTIONS)
    target_compile_options(
      ${fq_build_target_name}
      PRIVATE ${LIBC_UNITTEST_COMPILE_OPTIONS}
    )
  endif()
  if(NOT LIBC_UNITTEST_CXX_STANDARD)
    set(LIBC_UNITTEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set_target_properties(
    ${fq_build_target_name}
    PROPERTIES
      CXX_STANDARD ${LIBC_UNITTEST_CXX_STANDARD}
  )

  set(link_libraries ${link_object_files})
  # Test object files will depend on LINK_LIBRARIES passed down from `add_fp_unittest`
  foreach(lib IN LISTS LIBC_UNITTEST_LINK_LIBRARIES)
    if(TARGET ${lib}.unit)
      list(APPEND link_libraries ${lib}.unit)
    else()
      list(APPEND link_libraries ${lib})
    endif()
  endforeach()

  set_target_properties(${fq_build_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_build_target_name}
    ${fq_deps_list}
  )

  # LibcUnitTest should not depend on anything in LINK_LIBRARIES.
  list(APPEND link_libraries LibcDeathTestExecutors.unit LibcTest.unit)

  target_link_libraries(${fq_build_target_name} PRIVATE ${link_libraries})

  if(NOT LIBC_UNITTEST_NO_RUN_POSTBUILD)
    add_custom_target(
      ${fq_target_name}
      COMMAND ${fq_build_target_name}
      COMMENT "Running unit test ${fq_target_name}"
    )
  endif()

  if(LIBC_UNITTEST_SUITE)
    add_dependencies(
      ${LIBC_UNITTEST_SUITE}
      ${fq_target_name}
    )
  endif()
  add_dependencies(libc-unit-tests ${fq_target_name})
endfunction(create_libc_unittest)

# Internal function, used by `add_libc_unittest`.
function(expand_flags_for_libc_unittest target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "IGNORE_MARKER" # No Optional arguments
    "" # No Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    create_libc_unittest(
      ${target_name}
      DEPENDS "${EXPAND_FLAGS_DEPENDS}"
      FLAGS "${EXPAND_FLAGS_FLAGS}"
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
    return()
  endif()

  list(GET flags 0 flag)
  list(REMOVE_AT flags 0)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_libc_unittest(
      ${target_name}
      "${flags}"
      DEPENDS "${EXPAND_FLAGS_DEPENDS}" IGNORE_MARKER
      FLAGS "${EXPAND_FLAGS_FLAGS}" IGNORE_MARKER
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if("${modifier}" STREQUAL "")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_libc_unittest(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS "${NEW_DEPS}" IGNORE_MARKER
    FLAGS "${NEW_FLAGS}" IGNORE_MARKER
    "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
  )
endfunction(expand_flags_for_libc_unittest)

function(add_libc_unittest target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "" # Single value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})
  
  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Unit test ${fq_target_name} has FLAGS: ${flags}")
  endif()

  expand_flags_for_libc_unittest(
    ${fq_target_name}
    "${flags}"
    DEPENDS ${fq_deps_list} IGNORE_MARKER
    FLAGS ${flags} IGNORE_MARKER
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_libc_unittest)

function(add_libc_exhaustive_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(exhaustive-check-libc ${suite_name})
endfunction(add_libc_exhaustive_testsuite)

function(add_libc_long_running_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(libc-long-running-tests ${suite_name})
endfunction(add_libc_long_running_testsuite)

# Rule to add a fuzzer test.
# Usage
#    add_libc_fuzzer(
#      <target name>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#    )
function(add_libc_fuzzer target_name)
  cmake_parse_arguments(
    "LIBC_FUZZER"
    "NEED_MPFR" # Optional arguments
    "" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_FUZZER_SRCS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_FUZZER_DEPENDS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  list(APPEND LIBC_FUZZER_LINK_LIBRARIES "")
  if(LIBC_FUZZER_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message(VERBOSE "Fuzz test ${name} will be skipped as MPFR library is not available.")
      return()
    endif()
    list(APPEND LIBC_FUZZER_LINK_LIBRARIES mpfr gmp)
  endif()


  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${LIBC_FUZZER_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Skipping fuzzer target ${fq_target_name} as it has missing deps: "
              "${skipped_entrypoints_list}.")
      message(STATUS ${msg})
    endif()
    add_custom_target(${fq_target_name})

    # A post build custom command is used to avoid running the command always.
    add_custom_command(
      TARGET ${fq_target_name}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo ${msg}
    )
    return()
  endif()

  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_FUZZER_SRCS}
    ${LIBC_FUZZER_HDRS}
  )
  target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})

  target_link_libraries(${fq_target_name} PRIVATE 
    ${link_object_files} 
    ${LIBC_FUZZER_LINK_LIBRARIES}
  )

  set_target_properties(${fq_target_name}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )
  add_dependencies(libc-fuzzer ${fq_target_name})

  target_compile_options(${fq_target_name}
    PRIVATE
    ${LIBC_FUZZER_COMPILE_OPTIONS})

endfunction(add_libc_fuzzer)

# DEPRECATED: Use add_hermetic_test instead.
#
# Rule to add an integration test. An integration test is like a unit test
# but does not use the system libc. Not even the startup objects from the
# system libc are linked in to the final executable. The final exe is fully
# statically linked. The libc that the final exe links to consists of only
# the object files of the DEPENDS targets.
# 
# Usage:
#   add_integration_test(
#     <target name>
#     SUITE <the suite to which the test should belong>
#     SRCS <src1.cpp> [src2.cpp ...]
#     HDRS [hdr1.cpp ...]
#     DEPENDS <list of entrypoint or other object targets>
#     ARGS <list of command line arguments to be passed to the test>
#     ENV <list of environment variables to set before running the test>
#     COMPILE_OPTIONS <list of special compile options for this target>
#   )
#
# The DEPENDS list can be empty. If not empty, it should be a list of
# targets added with add_entrypoint_object or add_object_library.
function(add_integration_test test_name)
  get_fq_target_name(${test_name} fq_target_name)
  set(supported_targets gpu linux)
  if(NOT (${LIBC_TARGET_OS} IN_LIST supported_targets))
    message(STATUS "Skipping ${fq_target_name} as it is not available on ${LIBC_TARGET_OS}.")
    return()
  endif()
  cmake_parse_arguments(
    "INTEGRATION_TEST"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;HDRS;DEPENDS;ARGS;ENV;COMPILE_OPTIONS;LOADER_ARGS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT INTEGRATION_TEST_SUITE)
    message(FATAL_ERROR "SUITE not specified for ${fq_target_name}")
  endif()
  if(NOT INTEGRATION_TEST_SRCS)
    message(FATAL_ERROR "The SRCS list for add_integration_test is missing.")
  endif()
  if(NOT TARGET libc.startup.${LIBC_TARGET_OS}.crt1)
    message(FATAL_ERROR "The 'crt1' target for the integration test is missing.")
  endif()

  get_fq_target_name(${test_name}.libc fq_libc_target_name)

  get_fq_deps_list(fq_deps_list ${INTEGRATION_TEST_DEPENDS})
  list(APPEND fq_deps_list
      # All integration tests use the operating system's startup object with the
      # integration test object and need to inherit the same dependencies.
      libc.startup.${LIBC_TARGET_OS}.crt1
      libc.test.IntegrationTest.test
      # We always add the memory functions objects. This is because the
      # compiler's codegen can emit calls to the C memory functions.
      libc.src.string.bcmp
      libc.src.string.bzero
      libc.src.string.memcmp
      libc.src.string.memcpy
      libc.src.string.memmove
      libc.src.string.memset
  )
  list(REMOVE_DUPLICATES fq_deps_list)

  # TODO: Instead of gathering internal object files from entrypoints,
  # collect the object files with public names of entrypoints.
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Skipping unittest ${fq_target_name} as it has missing deps: "
              "${skipped_entrypoints_list}.")
      message(STATUS ${msg})
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
    # The NVIDIA 'nvlink' linker does not currently support static libraries.
    $<$<BOOL:${LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX}>:${link_object_files}>
    ${INTEGRATION_TEST_SRCS}
    ${INTEGRATION_TEST_HDRS}
  )
  set_target_properties(${fq_build_target_name}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  target_include_directories(${fq_build_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  target_compile_options(${fq_build_target_name}
      PRIVATE -fpie -ffreestanding -fno-exceptions -fno-rtti ${INTEGRATION_TEST_COMPILE_OPTIONS})
  # The GPU build requires overriding the default CMake triple and architecture.
  if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
    target_compile_options(${fq_build_target_name} PRIVATE
                           -nogpulib -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE}
                           -flto --target=${LIBC_GPU_TARGET_TRIPLE}
                           -mcode-object-version=${LIBC_GPU_CODE_OBJECT_VERSION})
  elseif(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
    get_nvptx_compile_options(nvptx_options ${LIBC_GPU_TARGET_ARCHITECTURE})
    target_compile_options(${fq_build_target_name} PRIVATE
                           -nogpulib ${nvptx_options} -fno-use-cxa-atexit
                           --target=${LIBC_GPU_TARGET_TRIPLE})
  endif()

  target_link_options(${fq_build_target_name} PRIVATE -nostdlib -static)
  target_link_libraries(
    ${fq_build_target_name}
    # The NVIDIA 'nvlink' linker does not currently support static libraries.
    $<$<NOT:$<BOOL:${LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX}>>:${fq_target_name}.__libc__>
    libc.startup.${LIBC_TARGET_OS}.crt1
    libc.test.IntegrationTest.test)
  add_dependencies(${fq_build_target_name}
                   libc.test.IntegrationTest.test
                   ${INTEGRATION_TEST_DEPENDS})

  # Tests on the GPU require an external loader utility to launch the kernel.
  if(TARGET libc.utils.gpu.loader)
    add_dependencies(${fq_build_target_name} libc.utils.gpu.loader)
    get_target_property(gpu_loader_exe libc.utils.gpu.loader "EXECUTABLE")
  endif()

  # We have to use a separate var to store the command as a list because
  # the COMMAND option of `add_custom_target` cannot handle empty vars in the
  # command. For example, if INTEGRATION_TEST_ENV is empty, the actual
  # command also will not run. So, we use this list and tell `add_custom_target`
  # to expand the list (by including the option COMMAND_EXPAND_LISTS). This
  # makes `add_custom_target` construct the correct command and execute it.
  set(test_cmd
      ${INTEGRATION_TEST_ENV}
      $<$<BOOL:${LIBC_TARGET_ARCHITECTURE_IS_GPU}>:${gpu_loader_exe}>
      ${INTEGRATION_TEST_LOADER_ARGS}
      $<TARGET_FILE:${fq_build_target_name}> ${INTEGRATION_TEST_ARGS})
  add_custom_target(
    ${fq_target_name}
    COMMAND ${test_cmd}
    COMMAND_EXPAND_LISTS
    COMMENT "Running integration test ${fq_target_name}"
  )
  add_dependencies(${INTEGRATION_TEST_SUITE} ${fq_target_name})
endfunction(add_integration_test)

set(LIBC_HERMETIC_TEST_COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_DEFAULT}
    -fpie -ffreestanding -fno-exceptions -fno-rtti)
# The GPU build requires overriding the default CMake triple and architecture.
if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  list(APPEND LIBC_HERMETIC_TEST_COMPILE_OPTIONS
       -nogpulib -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE} -flto
       --target=${LIBC_GPU_TARGET_TRIPLE}
       -mcode-object-version=${LIBC_GPU_CODE_OBJECT_VERSION})
elseif(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  get_nvptx_compile_options(nvptx_options ${LIBC_GPU_TARGET_ARCHITECTURE})
  list(APPEND LIBC_HERMETIC_TEST_COMPILE_OPTIONS
       -nogpulib ${nvptx_options} -fno-use-cxa-atexit --target=${LIBC_GPU_TARGET_TRIPLE})
endif()

# Rule to add a hermetic test. A hermetic test is one whose executable is fully
# statically linked and consists of pieces drawn only from LLVM's libc. Nothing,
# including the startup objects, come from the system libc.
#
# Usage:
#   add_libc_hermetic_test(
#     <target name>
#     SUITE <the suite to which the test should belong>
#     SRCS <src1.cpp> [src2.cpp ...]
#     HDRS [hdr1.cpp ...]
#     DEPENDS <list of entrypoint or other object targets>
#     ARGS <list of command line arguments to be passed to the test>
#     ENV <list of environment variables to set before running the test>
#     COMPILE_OPTIONS <list of special compile options for the test>
#     LINK_LIBRARIES <list of linking libraries for this target>
#     LOADER_ARGS <list of special args to loaders (like the GPU loader)>
#   )
function(add_libc_hermetic_test test_name)
  if(NOT TARGET libc.startup.${LIBC_TARGET_OS}.crt1)
    message(VERBOSE "Skipping ${fq_target_name} as it is not available on ${LIBC_TARGET_OS}.")
    return()
  endif()
  cmake_parse_arguments(
    "HERMETIC_TEST"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;HDRS;DEPENDS;ARGS;ENV;COMPILE_OPTIONS;LINK_LIBRARIES;LOADER_ARGS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT HERMETIC_TEST_SUITE)
    message(FATAL_ERROR "SUITE not specified for ${fq_target_name}")
  endif()
  if(NOT HERMETIC_TEST_SRCS)
    message(FATAL_ERROR "The SRCS list for add_integration_test is missing.")
  endif()

  get_fq_target_name(${test_name} fq_target_name)
  get_fq_target_name(${test_name}.libc fq_libc_target_name)

  get_fq_deps_list(fq_deps_list ${HERMETIC_TEST_DEPENDS})
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
    set(msg "Skipping unittest ${fq_target_name} as it has missing deps: "
            "${skipped_entrypoints_list}.")
    message(STATUS ${msg})
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
    # The NVIDIA 'nvlink' linker does not currently support static libraries.
    $<$<BOOL:${LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX}>:${link_object_files}>
    ${HERMETIC_TEST_SRCS}
    ${HERMETIC_TEST_HDRS}
  )
  set_target_properties(${fq_build_target_name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      #OUTPUT_NAME ${fq_target_name}
  )
  target_include_directories(${fq_build_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  target_compile_options(${fq_build_target_name}
      PRIVATE ${LIBC_HERMETIC_TEST_COMPILE_OPTIONS} ${HERMETIC_TEST_COMPILE_OPTIONS})

  set(link_libraries "")
  foreach(lib IN LISTS HERMETIC_TEST_LINK_LIBRARIES)
    if(TARGET ${lib}.hermetic)
      list(APPEND link_libraries ${lib}.hermetic)
    else()
      list(APPEND link_libraries ${lib})
    endif()
  endforeach()

  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    target_link_options(${fq_build_target_name} PRIVATE -nostdlib -static)
  else()
    target_link_options(${fq_build_target_name} PRIVATE -nolibc -nostartfiles -nostdlib++ -static)
  endif()
  target_link_libraries(
    ${fq_build_target_name}
    PRIVATE
      libc.startup.${LIBC_TARGET_OS}.crt1
      ${link_libraries}
      LibcTest.hermetic
      LibcHermeticTestSupport.hermetic
      # The NVIDIA 'nvlink' linker does not currently support static libraries.
      $<$<NOT:$<BOOL:${LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX}>>:${fq_target_name}.__libc__>)
  add_dependencies(${fq_build_target_name}
                   LibcTest.hermetic
                   libc.test.UnitTest.ErrnoSetterMatcher
                   ${fq_deps_list})

  # Tests on the GPU require an external loader utility to launch the kernel.
  if(TARGET libc.utils.gpu.loader)
    add_dependencies(${fq_build_target_name} libc.utils.gpu.loader)
    get_target_property(gpu_loader_exe libc.utils.gpu.loader "EXECUTABLE")
  endif()

  set(test_cmd ${HERMETIC_TEST_ENV}
      $<$<BOOL:${LIBC_TARGET_ARCHITECTURE_IS_GPU}>:${gpu_loader_exe}> ${HERMETIC_TEST_LOADER_ARGS}
      $<TARGET_FILE:${fq_build_target_name}> ${HERMETIC_TEST_ARGS})
  add_custom_target(
    ${fq_target_name}
    COMMAND ${test_cmd}
    COMMAND_EXPAND_LISTS
    COMMENT "Running hermetic test ${fq_target_name}"
    ${LIBC_HERMETIC_TEST_JOB_POOL}
  )

  add_dependencies(${HERMETIC_TEST_SUITE} ${fq_target_name})
  add_dependencies(libc-hermetic-tests ${fq_target_name})
endfunction(add_libc_hermetic_test)

# A convenience function to add both a unit test as well as a hermetic test.
function(add_libc_test test_name)
  cmake_parse_arguments(
    "LIBC_TEST"
    "UNIT_TEST_ONLY;HERMETIC_TEST_ONLY" # Optional arguments
    "" # Single value arguments
    "" # Multi-value arguments
    ${ARGN}
  )
  if(LIBC_ENABLE_UNITTESTS AND NOT LIBC_TEST_HERMETIC_TEST_ONLY)
    add_libc_unittest(${test_name}.__unit__ ${LIBC_TEST_UNPARSED_ARGUMENTS})
  endif()
  if(LIBC_ENABLE_HERMETIC_TESTS AND NOT LIBC_TEST_UNIT_TEST_ONLY)
    add_libc_hermetic_test(${test_name}.__hermetic__ ${LIBC_TEST_UNPARSED_ARGUMENTS})
    get_fq_target_name(${test_name} fq_test_name)
    if(TARGET ${fq_test_name}.__hermetic__ AND TARGET ${fq_test_name}.__unit__)
      # Tests like the file tests perform file operations on disk file. If we
      # don't chain up the unit test and hermetic test, then those tests will
      # step on each other's files.
      add_dependencies(${fq_test_name}.__hermetic__ ${fq_test_name}.__unit__)
    endif()
  endif()
endfunction(add_libc_test)
