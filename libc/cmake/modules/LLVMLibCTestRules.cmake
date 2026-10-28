if(LIBC_TARGET_OS_IS_LINUX OR LIBC_TARGET_OS_IS_DARWIN)
  set(LIBC_TEST_SUBPROCESS_TESTS 1)
else()
  set(LIBC_TEST_SUBPROCESS_TESTS 0)
endif()

function(_get_common_test_compile_options_impl output_var c_test)
  set(config_flags ${LIBC_COMPILE_OPTIONS_FROM_CONFIG})
  set(arch_flags ${LIBC_COMPILE_OPTIONS_FROM_ARCH})

  # Tests guarded by LIBC_ADD_NULL_CHECKS use EXPECT_DEATH / ASSERT_DEATH to
  # verify that passing nullptr triggers a crash (__builtin_trap). Executing
  # death tests requires spawning a subprocess via LibcDeathTestExecutors
  # (fork, pipe, poll, waitpid, kill, etc.), which is currently only supported
  # on Linux. In full-build mode, "-DLIBC_ADD_NULL_CHECKS" was historically
  # stripped (#142085) before LibcDeathTestExecutors was ported to hermetic
  # mode (#211484).
  # TODO: Once LibcDeathTestExecutors is unified with add_object_library and
  # transitive DAG dependency resolution is in place, re-enable
  # LIBC_ADD_NULL_CHECKS for full-build unit tests where LibcDeathTestExecutors
  # and its dependencies are available.
  if(NOT ${LIBC_TARGET_OS} STREQUAL "linux" OR LLVM_LIBC_FULL_BUILD)
    if(MSVC)
      list(REMOVE_ITEM config_flags "/DLIBC_ADD_NULL_CHECKS")
    else()
      list(REMOVE_ITEM config_flags "-DLIBC_ADD_NULL_CHECKS")
    endif()
  endif()

  set(compile_options
    ${config_flags}
    ${arch_flags})
  libc_add_definition(compile_options
    "LIBC_TEST_FLOAT_RANGE_COUNT=${LIBC_TEST_FLOAT_RANGE_COUNT}")

  libc_add_definition(compile_options
    "LIBC_TEST_SUBPROCESS_TESTS=${LIBC_TEST_SUBPROCESS_TESTS}")

  if(LIBC_TEST_MAX_CONCURRENCY)
    libc_add_definition(compile_options
      "LIBC_TEST_MAX_CONCURRENCY=${LIBC_TEST_MAX_CONCURRENCY}")
  endif()

  if(LIBC_TEST_SUBPROCESS_TESTS)
    # EXPECT_DEATH and ASSERT_DEATH might be quite slow.  LIBC_TEST_SKIP_DEATH_TESTS
    # will make those tests no-op to reduce the overall test time.
    if(LIBC_TEST_SKIP_DEATH_TESTS)
      if(LIBC_CMAKE_VERBOSE_LOGGING)
        message(STATUS "LIBC_TEST_SKIP_DEATH_TESTS is set.  EXPECT_DEATH/ASSERT_DEATH are no-op.")
      endif()
      list(APPEND compile_options "-DLIBC_TEST_SKIP_DEATH_TESTS")
    endif()
  endif()

  if(CMAKE_CROSSCOMPILING_EMULATOR)
    list(APPEND compile_options "-DLIBC_TEST_UNDER_EMULATOR")
  endif()

  if(LLVM_LIBC_COMPILER_IS_GCC_COMPATIBLE)
    list(APPEND compile_options "-fpie")

    if(LLVM_LIBC_FULL_BUILD)
      list(APPEND compile_options "-DLIBC_FULL_BUILD")
      # Only add -ffreestanding flag in full build mode.
      list(APPEND compile_options "-ffreestanding")
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        (LIBC_TARGET_ARCHITECTURE_IS_X86_64 OR LIBC_TARGET_ARCHITECTURE_IS_X86))
        list(APPEND compile_options "-D_MM_MALLOC_H_INCLUDED")
      endif()
      list(APPEND compile_options "-fno-exceptions")
      list(APPEND compile_options "-fno-unwind-tables")
      list(APPEND compile_options "-fno-asynchronous-unwind-tables")
      if(NOT c_test)
        list(APPEND compile_options "-fno-rtti")
      endif()
      # The GPU build requires overriding the default CMake triple and architecture.
      if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
        list(APPEND compile_options
          -Wno-multi-gpu -nogpulib -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE} -flto
          -mcode-object-version=${LIBC_GPU_CODE_OBJECT_VERSION})
      elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
        list(APPEND compile_options
          -Wno-multi-gpu --cuda-path=${LIBC_CUDA_ROOT}
          -nogpulib -march=${LIBC_GPU_TARGET_ARCHITECTURE} -fno-use-cxa-atexit)
      endif()
    endif()

    if(LIBC_COMPILER_HAS_FIXED_POINT)
      list(APPEND compile_options "-ffixed-point")
    endif()

    list(APPEND compile_options "-Wall")
    list(APPEND compile_options "-Wextra")
    # -DLIBC_WNO_ERROR=ON if you can't build cleanly with -Werror.
    if(NOT LIBC_WNO_ERROR)
    # list(APPEND compile_options "-Werror")
    endif()
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0.0"))
      list(APPEND compile_options "-Wconversion")
    else()
      list(APPEND compile_options "-Wno-type-limits")
      list(APPEND compile_options "-Wno-attributes")
    endif()
    # FIXME: convert to -Wsign-conversion
    list(APPEND compile_options "-Wno-sign-conversion")
    list(APPEND compile_options "-Wimplicit-fallthrough")
    list(APPEND compile_options "-Wwrite-strings")
    # Silence this warning because _Complex is a part of C99.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      if(NOT c_test)
        list(APPEND compile_options "-fext-numeric-literals")
      endif()
    else()
      list(APPEND compile_options "-Wno-c99-extensions")
      list(APPEND compile_options "-Wno-gnu-imaginary-constant")
    endif()
    list(APPEND compile_options "-Wno-pedantic")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      list(APPEND compile_options "-Wstrict-prototypes")
      list(APPEND compile_options "-Wextra-semi")
      list(APPEND compile_options "-Wnewline-eof")
      list(APPEND compile_options "-Wnonportable-system-include-path")
      list(APPEND compile_options "-Wthread-safety")
    endif()

    if(LIBC_COMPILER_HAS_STDC_FENV_ACCESS)
      list(APPEND compile_options "-DLIBC_COMPILER_HAS_STDC_FENV_ACCESS")
    endif()
  endif()
  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction()
_get_common_test_compile_options_impl(LIBC_COMMON_TEST_COMPILE_OPTIONS_CPP FALSE)
_get_common_test_compile_options_impl(LIBC_COMMON_TEST_COMPILE_OPTIONS_C TRUE)

function(_get_common_test_compile_options output_var c_test flags)
  _get_compile_options_from_flags(compile_flags ${flags})
  if(c_test)
    set(common_test_opts ${LIBC_COMMON_TEST_COMPILE_OPTIONS_C})
  else()
    set(common_test_opts ${LIBC_COMMON_TEST_COMPILE_OPTIONS_CPP})
  endif()
  set(${output_var}
    ${LIBC_COMPILE_OPTIONS_DEFAULT}
    ${LIBC_TEST_COMPILE_OPTIONS_DEFAULT}
    ${compile_flags}
    ${common_test_opts}
    PARENT_SCOPE)
endfunction()

function(get_link_options link_options)
  set(link_opts "")
  foreach(opt IN LISTS ARGN)
    if((NOT ${opt} MATCHES "^/D") AND (NOT ${opt} MATCHES "^-D"))
      list(APPEND link_opts ${opt})
    endif()
  endforeach()
  set(${link_options} "${link_opts}" PARENT_SCOPE)
endfunction()

function(_get_common_test_link_options_impl output_var)
  if(LLVM_LIBC_FULL_BUILD)
    if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
      set(link_options
        ${LIBC_COMPILE_OPTIONS_DEFAULT}
        ${LIBC_LINK_OPTIONS_DEFAULT}
        ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
        -Wno-multi-gpu
        -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE}
        -flto
        -nostdlib
        -static
        "-Wl,-mllvm,-amdhsa-code-object-version=${LIBC_GPU_CODE_OBJECT_VERSION}"
      )
    elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
      set(link_options
        ${LIBC_COMPILE_OPTIONS_DEFAULT}
        ${LIBC_LINK_OPTIONS_DEFAULT}
        ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
        -Wno-multi-gpu
        "-Wl,--suppress-stack-size-warning"
        "-Wl,-mllvm,-nvptx-emit-init-fini-kernel"
        -march=${LIBC_GPU_TARGET_ARCHITECTURE}
        -nostdlib
        -static
        "--cuda-path=${LIBC_CUDA_ROOT}"
      )
    elseif(LIBC_CC_SUPPORTS_NOSTDLIBPP)
      set(link_options
        -nostartfiles
        -nodefaultlibs
        -nostdlib++
        -static
        ${LIBC_COMPILE_OPTIONS_DEFAULT}
        ${LIBC_LINK_OPTIONS_DEFAULT}
        ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
      )
      if(LIBC_ENABLE_COVERAGE)
        list(APPEND link_options
          -noprofilelib
          -u__llvm_profile_runtime
        )
      endif()
    else()
      # Older version of gcc does not support `nostdlib++` flag.  We use
      # `nostdlib` and link against libgcc_s, which cannot be linked statically.
      set(link_options
        -nostartfiles
        -nodefaultlibs
        -nostdlib
        ${LIBC_COMPILE_OPTIONS_DEFAULT}
        ${LIBC_LINK_OPTIONS_DEFAULT}
        ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
      )
      if(LIBC_ENABLE_COVERAGE)
        list(APPEND link_options
          -noprofilelib
          -u__llvm_profile_runtime
        )
      endif()
    endif()
  else()
    set(link_options
      ${LIBC_COMPILE_OPTIONS_DEFAULT}
      ${LIBC_TEST_COMPILE_OPTIONS_DEFAULT}
      ${LIBC_LINK_OPTIONS_DEFAULT}
      ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
    )
    if(LLVM_LIBC_COMPILER_IS_GCC_COMPATIBLE)
      list(APPEND link_options "-fpie")
    endif()
  endif()
  get_link_options(link_options ${link_options})
  set(${output_var} ${link_options} PARENT_SCOPE)
endfunction()
_get_common_test_link_options_impl(LIBC_COMMON_TEST_LINK_OPTIONS)

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
  set(checked_list "")
  set(unchecked_list "${ARGN}")
  list(REMOVE_DUPLICATES unchecked_list)

  foreach(dep IN LISTS unchecked_list)
    if(NOT TARGET ${dep})
      # Skip tests with undefined dependencies.
      # Compiler-RT targets are added only if they are enabled. However, such targets may not be defined
      # at the time of the libc build. We should skip checking such targets.
      if(NOT ${dep} MATCHES "^RTScudo.*|^RTGwp.*")
        list(APPEND skipped_list ${dep})
      endif()
      continue()
    endif()
    get_target_property(aliased_target ${dep} "ALIASED_TARGET")
    if(aliased_target)
      # If the target is just an alias, switch to the real target.
      set(dep ${aliased_target})
    endif()

    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT dep_type)
      # Skip tests with no object dependencies.
      continue()
    endif()

    get_target_property(dep_checked ${dep} "CHECK_OBJ_FOR_TESTS")

    if(dep_checked)
      # Target full dependency has already been checked.  Just use the results.
      get_target_property(dep_obj ${dep} "OBJECT_FILES_FOR_TESTS")
      get_target_property(dep_skip ${dep} "SKIPPED_LIST_FOR_TESTS")
    else()
      # Target full dependency hasn't been checked.  Recursively check its DEPS.
      set(dep_obj "${dep}")
      set(dep_skip "")

      get_target_property(indirect_deps ${dep} "DEPS")
      get_object_files_for_test(dep_obj dep_skip ${indirect_deps})

      if(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
        get_target_property(dep_object_files ${dep} "OBJECT_FILES")
        if(dep_object_files)
          list(APPEND dep_obj ${dep_object_files})
        endif()
      elseif(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
        get_target_property(is_skipped ${dep} "SKIPPED")
        if(is_skipped)
          list(APPEND dep_skip ${dep})
          list(REMOVE_ITEM dep_obj ${dep})
        endif()
        get_target_property(object_file_raw ${dep} "OBJECT_FILE_RAW")
        if(object_file_raw)
          # TODO: Remove this once we stop suffixing the target with ".__internal__"
          if(fq_target_name STREQUAL "libc.test.include.issignaling_c_test" OR fq_target_name STREQUAL "libc.test.include.iscanonical_c_test")
            string(REPLACE ".__internal__" "" object_file_raw ${object_file_raw})
          endif()
          # Adding libc.src.stdlib.exit normally contributes the internal entrypoint
          # object. External baremetal startup code can reference the public C exit
          # symbol, so use the public-packaging object for exit in this configuration.
          if(dep STREQUAL "libc.src.stdlib.exit" AND
            NOT LLVM_LIBC_HERMETIC_TEST_USE_INTERNAL_STARTUP AND
            LIBC_TARGET_OS_IS_BAREMETAL)
            string(REPLACE ".__internal__" "" object_file_raw ${object_file_raw})
          endif()
          list(APPEND dep_obj ${object_file_raw})
        endif()
      endif()

      set_target_properties(${dep} PROPERTIES
        OBJECT_FILES_FOR_TESTS "${dep_obj}"
        SKIPPED_LIST_FOR_TESTS "${dep_skip}"
        CHECK_OBJ_FOR_TESTS "YES"
      )

    endif()

    list(APPEND object_files ${dep_obj})
    list(APPEND skipped_list ${dep_skip})

  endforeach()

  list(REMOVE_DUPLICATES object_files)
  set(${result} ${object_files} PARENT_SCOPE)
  list(REMOVE_DUPLICATES skipped_list)
  set(${skipped_entrypoints_list} ${skipped_list} PARENT_SCOPE)

endfunction()

# Get libgcc_s to be used in hermetic and integration tests.
if(NOT MSVC AND NOT LIBC_CC_SUPPORTS_NOSTDLIBPP)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc_s.so.1
    OUTPUT_VARIABLE LIBGCC_S_LOCATION)
  string(STRIP ${LIBGCC_S_LOCATION} LIBGCC_S_LOCATION)
endif()

# Hermetic and integration tests link with -nodefaultlibs, so the compiler
# runtime builtins have to be linked explicitly. -print-libgcc-file-name
# reports whichever library the driver would use (libgcc.a or
# libclang_rt.builtins.a). The query uses the same flags as the test link so
# that the driver selects the same multilib. GPU and UEFI targets do not link a
# compiler runtime.
if(LLVM_INCLUDE_TESTS AND LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_OS_IS_GPU AND
  NOT LIBC_TARGET_OS_IS_UEFI AND
  NOT LIBC_TEST_BUILTINS_LIBRARY AND NOT LIBC_TEST_BUILTINS_TARGET)
  separate_arguments(libc_cxx_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} ${libc_cxx_flags}
      ${LIBC_COMPILE_OPTIONS_DEFAULT} ${LIBC_TEST_LINK_OPTIONS_DEFAULT}
      -print-libgcc-file-name
    OUTPUT_VARIABLE LIBC_TEST_BUILTINS_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT EXISTS "${LIBC_TEST_BUILTINS_LIBRARY}")
    message(FATAL_ERROR "Compiler runtime builtins library "
      "'${LIBC_TEST_BUILTINS_LIBRARY}' not found. Set "
      "LIBC_TEST_BUILTINS_LIBRARY or LIBC_TEST_BUILTINS_TARGET.")
  endif()
endif()

# Rule to add a test suite in the hierarchy.
# Usage:
#   add_libc_testsuite(
#     <suite_name>
#     [PARENT <parent_suite> | ROOT]
#     [BUILD_ONLY]
#   )
# Automatically creates <suite_name> (run target) and <suite_name>-build (build target),
# and connects them to <parent_suite> (defaulting to libc-tests unless ROOT is specified).
function(add_libc_testsuite suite_name)
  llvm_libc_parse_arguments(SUITE ${ARGN})

  if(SUITE_ROOT AND SUITE_PARENT)
    message(FATAL_ERROR
      "Cannot specify both ROOT and PARENT for test suite ${suite_name}.")
  endif()

  add_custom_target(${suite_name})
  add_custom_target(${suite_name}-build)

  if(SUITE_BUILD_ONLY)
    set_target_properties(${suite_name} PROPERTIES BUILD_ONLY TRUE)
  endif()

  if(NOT SUITE_ROOT)
    if(SUITE_PARENT)
      set(parent_suite ${SUITE_PARENT})
    else()
      set(parent_suite libc-tests)
    endif()
    if(SUITE_BUILD_ONLY)
      add_dependencies(${parent_suite} ${suite_name}-build)
    else()
      add_dependencies(${parent_suite} ${suite_name})
    endif()
    add_dependencies(${parent_suite}-build ${suite_name}-build)
  endif()
endfunction()

# Helper function for create_libc_test: returns the list of libc entrypoints
# needed to support code coverage profiling in hermetic/full-build mode.
function(_get_libc_coverage_dependencies output_var)
  set(${output_var}
    libc.src.stdio.fclose
    libc.src.stdio.fdopen
    libc.src.stdio.feof
    libc.src.stdio.fflush
    libc.src.stdio.fileno
    libc.src.stdio.fopen
    libc.src.stdio.fprintf
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.ftell
    libc.src.stdio.fwrite
    libc.src.stdio.snprintf
    libc.src.stdio.stderr
    libc.src.stdio.vfprintf
    libc.src.stdio.vsnprintf
    libc.src.fcntl.fcntl
    libc.src.fcntl.open
    libc.src.stdlib.getenv
    libc.src.stdlib.setenv
    libc.src.stdlib.strtol
    libc.src.string.strchr
    libc.src.string.strcmp
    libc.src.string.strdup
    libc.src.string.strerror
    libc.src.string.strlen
    libc.src.string.strncpy
    libc.src.string.strrchr
    libc.src.sys.mman.madvise
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap
    libc.src.sys.prctl.prctl
    libc.src.sys.stat.mkdir
    libc.src.sys.utsname.uname
    libc.src.unistd.fork
    libc.src.unistd.ftruncate
    libc.src.unistd.getpagesize
    libc.src.unistd.getpid
    PARENT_SCOPE
  )
endfunction()

# Helper function for create_libc_test: collects fully qualified dependencies
# for a test target, including base test framework libraries, hermetic runtime
# entrypoints, subprocess support, and coverage dependencies.
function(_get_libc_test_dependencies output_var user_depends startup_target_dep is_c_test is_fp_test is_integration_test)
  get_fq_deps_list(fq_deps_list ${user_depends})
  if(is_integration_test)
    list(APPEND fq_deps_list
      generate-libc-headers
      libc.test.IntegrationTest.test
    )
  else()
    list(APPEND fq_deps_list
      libc.src.__support.StringUtil.error_to_string
      libc.test.UnitTest.ErrnoSetterMatcher
      libc.test.UnitTest.LibcTest
    )
    if(is_c_test)
      list(APPEND fq_deps_list libc.test.UnitTest.LibcCTest)
    else()
      list(APPEND fq_deps_list libc.test.UnitTest.LibcDeathTestExecutors)
    endif()
    if(is_fp_test)
      list(APPEND fq_deps_list
        libc.hdr.math_macros
        libc.hdr.stdint_proxy
        libc.test.UnitTest.FPTestHelpers
      )
    endif()
  endif()

  if(LLVM_LIBC_FULL_BUILD)
    list(APPEND fq_deps_list
      ${startup_target_dep}
      libc.src.string.memcmp
      libc.src.string.memcpy
      libc.src.string.memmove
      libc.src.string.memset
      libc.src.strings.bcmp
      libc.src.strings.bzero
    )
    if(libc.src.compiler.__stack_chk_fail IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
      list(APPEND fq_deps_list libc.src.compiler.__stack_chk_fail)
    endif()
    if(NOT is_integration_test)
      list(APPEND fq_deps_list libc.test.UnitTest.HermeticTestUtils)
      if(LIBC_TARGET_ARCHITECTURE_IS_AARCH64 AND NOT(LIBC_TARGET_OS_IS_BAREMETAL))
        list(APPEND fq_deps_list libc.src.sys.auxv.getauxval)
      endif()
      if(LIBC_TEST_SUBPROCESS_TESTS AND NOT is_c_test)
        list(APPEND fq_deps_list
          libc.src.poll.poll
          libc.src.signal.kill
          libc.src.signal.signal
          libc.src.stdio.fflush
          libc.src.stdio.stderr
          libc.src.stdio.stdout
          libc.src.stdlib.exit
          libc.src.string.strsignal
          libc.src.sys.wait.waitpid
          libc.src.unistd.close
          libc.src.unistd.fork
          libc.src.unistd.pipe
        )
      endif()
      if(NOT LLVM_LIBC_HERMETIC_TEST_USE_INTERNAL_STARTUP AND LIBC_TARGET_OS_IS_BAREMETAL)
        # External baremetal startup objects still need libc's process startup and
        # termination entrypoints, even when the test itself does not depend on them.
        list(APPEND fq_deps_list
          libc.src.stdlib.atexit
          libc.src.stdlib.exit
          libc.startup.baremetal.init
        )
      endif()
      if(libc.src.time.clock IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
        list(APPEND fq_deps_list libc.src.time.clock)
      endif()
    endif()
    if(LIBC_ENABLE_COVERAGE)
      _get_libc_coverage_dependencies(coverage_deps)
      list(APPEND fq_deps_list ${coverage_deps})
    endif()
  endif()
  list(REMOVE_DUPLICATES fq_deps_list)
  set(${output_var} ${fq_deps_list} PARENT_SCOPE)
endfunction()

# Helper function for create_libc_test: configures linking for full-build hermetic
# tests by creating the static .__libc__ archive and wiring link libraries and
# compiler runtime.
function(_link_libc_hermetic_test fq_target_name fq_build_target_name startup_target_dep link_object_files test_link_libraries)
  if(LIBC_ENABLE_COVERAGE)
    _get_libc_coverage_dependencies(coverage_deps)
    foreach(cov_dep IN LISTS coverage_deps)
      get_target_property(is_alias ${cov_dep} "IS_ALIAS")
      if(is_alias)
        get_target_property(real_target ${cov_dep} "DEPS")
      else()
        set(real_target ${cov_dep})
      endif()
      string(REPLACE "${real_target}.__internal__" "${real_target}" link_object_files "${link_object_files}")
    endforeach()
  endif()

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

  set(compiler_runtime "")
  if(NOT LIBC_TARGET_ARCHITECTURE_IS_AMDGPU AND
    NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX AND
    NOT LIBC_CC_SUPPORTS_NOSTDLIBPP)
    list(APPEND compiler_runtime ${LIBGCC_S_LOCATION})
  endif()

  set(coverage_link_libs "")
  if(LIBC_ENABLE_COVERAGE)
    set(coverage_link_libs
      "${LIBC_CLANG_PROFILE_LIB}"
      ${fq_target_name}.__libc__
    )
  endif()

  target_link_libraries(
    ${fq_build_target_name}
    PRIVATE
      ${startup_target_dep}
      ${test_link_libraries}
      ${fq_target_name}.__libc__
      ${coverage_link_libs}
      ${LIBC_TEST_BUILTINS_LIBRARY}
      ${compiler_runtime}
  )
  if(LIBC_TEST_BUILTINS_TARGET)
    target_link_libraries(${fq_build_target_name} PRIVATE $<TARGET_FILE:${LIBC_TEST_BUILTINS_TARGET}>)
    add_dependencies(${fq_build_target_name} ${LIBC_TEST_BUILTINS_TARGET})
  endif()
endfunction()

# Helper function for create_libc_test: sets up the post-build custom target
# and command to run the test binary (handling emulators, argument passing,
# and parameter files).
function(_add_libc_test_runner fq_target_name fq_build_target_name test_args loader_args test_env)
  set(test_binary "${CMAKE_CURRENT_BINARY_DIR}/${fq_build_target_name}${CMAKE_EXECUTABLE_SUFFIX}")
  if(LIBC_TEST_CMD)
    string(REPLACE "@BINARY@" "${test_binary}" test_cmd_parsed ${LIBC_TEST_CMD})
    string(REPLACE " " ";" test_cmd "${test_cmd_parsed}")
  else()
    set(test_cmd ${test_env})
    if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
      list(APPEND test_cmd "LIBOMPTARGET_STACK_SIZE=3072")
    endif()
    list(APPEND test_cmd
      ${CMAKE_CROSSCOMPILING_EMULATOR} ${loader_args}
      "${test_binary}" ${test_args})
  endif()

  if(loader_args OR test_args OR test_env OR LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
    set(_params_content "")
    foreach(_arg IN LISTS loader_args)
      string(APPEND _params_content "${_arg}\n")
    endforeach()
    string(APPEND _params_content "---\n")
    foreach(_arg IN LISTS test_args)
      string(APPEND _params_content "${_arg}\n")
    endforeach()
    string(APPEND _params_content "---\n")
    foreach(_env_entry IN LISTS test_env)
      string(APPEND _params_content "${_env_entry}\n")
    endforeach()
    if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
      string(APPEND _params_content "LIBOMPTARGET_STACK_SIZE=3072\n")
    endif()

    file(GENERATE
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fq_build_target_name}.params"
      CONTENT "${_params_content}"
    )
  endif()

  add_custom_target(
    ${fq_target_name}
    COMMAND ${test_cmd}
    COMMAND_EXPAND_LISTS
    DEPENDS ${fq_build_target_name}
    COMMENT "Running test ${fq_target_name}"
    ${LIBC_HERMETIC_TEST_JOB_POOL}
  )
endfunction()

# Helper function for create_libc_test: registers the test and its build target
# with the appropriate test suite.
function(_register_libc_test_with_suite fq_target_name fq_build_target_name suite build_only)
  if(build_only)
    add_dependencies(${suite} ${fq_build_target_name})
  else()
    add_dependencies(${suite} ${fq_target_name})
  endif()
  add_dependencies(${suite}-build ${fq_build_target_name})
endfunction()

# Core rule to create a libc test executable and runner (overlay or hermetic).
function(create_libc_test fq_target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  llvm_libc_parse_arguments(TEST ${ARGN})

  if(NOT TEST_SUITE)
    message(FATAL_ERROR "SUITE not specified for ${fq_target_name}.")
  endif()
  get_target_property(suite_build_only ${TEST_SUITE} BUILD_ONLY)

  # 1. Validate build mode, target platform, and external library requirements
  if(TEST_INTEGRATION_TEST)
    set(supported_targets gpu linux)
    if(NOT (${LIBC_TARGET_OS} IN_LIST supported_targets))
      message(STATUS "Skipping ${fq_target_name} as it is not available on ${LIBC_TARGET_OS}.")
      return()
    endif()
    set(TEST_FULL_BUILD_ONLY TRUE)
  endif()

  if(TEST_NEED_MPC)
    set(TEST_NEED_MPFR TRUE)
    if(NOT LIBC_TESTS_CAN_USE_MPC)
      message(VERBOSE "Complex test ${fq_target_name} will be skipped as MPC library is not available.")
      return()
    endif()
    list(APPEND TEST_LINK_LIBRARIES libc.utils.MPCWrapper)
  endif()

  if(TEST_NEED_MPFR_F80)
    set(TEST_NEED_MPFR TRUE)
    if(NOT LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
      message(VERBOSE "Math test ${fq_target_name} will be skipped as float80 support in MPFR is not available.")
      return()
    endif()
  endif()

  # TODO: To be removed when we find a workaround to run MPFR tests for
  # emulated type where compiler doesn't support the native float128
  # TODO: Check mpfr_buildopt_float128_p() returns non-zero.
  if(TEST_NEED_MPFR_F128)
    set(TEST_NEED_MPFR TRUE)
    if(NOT LIBC_TYPES_HAS_NATIVE_FLOAT128)
      message(VERBOSE "Math test ${fq_target_name} will be skipped as native float128 support in MPFR is not available.")
      return()
    endif()
  endif()

  if(TEST_NEED_MPFR)
    if(TEST_FULL_BUILD_ONLY)
      message(FATAL_ERROR "Full-build test cannot require MPFR.")
    endif()
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message(VERBOSE "Test ${fq_target_name} will be skipped as MPFR library is not available.")
      return()
    endif()
    set(TEST_OVERLAY_ONLY TRUE)
    list(APPEND TEST_LINK_LIBRARIES libc.utils.MPFRWrapper)
  endif()

  if(LLVM_LIBC_FULL_BUILD AND TEST_OVERLAY_ONLY)
    return()
  endif()
  if(NOT LLVM_LIBC_FULL_BUILD AND TEST_FULL_BUILD_ONLY)
    return()
  endif()

  set(startup_target_dep "")
  if(LLVM_LIBC_FULL_BUILD)
    set(startup_target libc.startup.${LIBC_TARGET_OS}.crt1)
    if(LLVM_LIBC_HERMETIC_TEST_USE_INTERNAL_STARTUP)
      if(TEST_STARTUP)
        message(FATAL_ERROR
          "Hermetic test ${fq_target_name} provides an external STARTUP target "
          "while internal startup is enabled.")
      endif()
      if(NOT TARGET ${startup_target})
        if(LIBC_CMAKE_VERBOSE_LOGGING)
          message(STATUS "Skipping ${fq_target_name} as ${startup_target} is not available on ${LIBC_TARGET_OS}.")
        endif()
        return()
      endif()
      set(startup_target_dep ${startup_target})
    elseif(TEST_STARTUP)
      get_fq_dep_name(startup_target_dep ${TEST_STARTUP})
      if(NOT TARGET ${startup_target_dep})
        message(FATAL_ERROR
          "Hermetic test ${fq_target_name} startup target ${startup_target_dep} does not exist.")
      endif()
    endif()

    if(NOT startup_target_dep AND NOT LIBC_TEST_LINK_OPTIONS_DEFAULT)
      if(LIBC_CMAKE_VERBOSE_LOGGING)
        message(STATUS "Skipping ${fq_target_name} as it has no startup provider.")
      endif()
      return()
    endif()
  endif()

  if(NOT TEST_SRCS)
    message(FATAL_ERROR "'create_libc_test' target requires a SRCS list of .cpp files.")
  endif()

  # 2. Resolve dependencies and link objects
  _get_libc_test_dependencies(
    fq_deps_list
    "${TEST_DEPENDS}"
    "${startup_target_dep}"
    "${TEST_C_TEST}"
    "${TEST_FP_TEST}"
    "${TEST_INTEGRATION_TEST}"
  )

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding test ${fq_target_name}")
    if("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS")
      foreach(dep IN LISTS TEST_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  get_object_files_for_test(
    link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Skipping test ${fq_target_name} as it has missing deps: "
        "${skipped_entrypoints_list}.")
      message(STATUS ${msg})
    endif()
    return()
  endif()
  list(REMOVE_DUPLICATES link_object_files)
  if(TEST_LINK_LIBRARIES)
    list(REMOVE_DUPLICATES TEST_LINK_LIBRARIES)
  endif()

  # 3. Create the test executable
  if(TEST_BUILD_ONLY OR suite_build_only)
    set(fq_build_target_name ${fq_target_name}.__build_only__)
  else()
    set(fq_build_target_name ${fq_target_name}.__build__)
  endif()

  add_executable(
    ${fq_build_target_name}
    EXCLUDE_FROM_ALL
    ${TEST_SRCS}
    ${TEST_HDRS}
  )

  target_include_directories(${fq_build_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  target_include_directories(${fq_build_target_name} PRIVATE ${LIBC_BUILD_DIR})

  _get_common_test_compile_options(compile_options "${TEST_C_TEST}" "${TEST_FLAGS}")
  target_compile_options(${fq_build_target_name} PRIVATE ${compile_options} ${TEST_COMPILE_OPTIONS})

  if(NOT TEST_CXX_STANDARD)
    set(TEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set_target_properties(
    ${fq_build_target_name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      CXX_STANDARD ${TEST_CXX_STANDARD}
  )

  # 4. Configure linking
  target_link_options(
    ${fq_build_target_name}
    PRIVATE
    ${LIBC_COMMON_TEST_LINK_OPTIONS}
    ${TEST_LINK_OPTIONS}
  )
  if(LLVM_LIBC_FULL_BUILD)
    _link_libc_hermetic_test(
      ${fq_target_name}
      ${fq_build_target_name}
      "${startup_target_dep}"
      "${link_object_files}"
      "${TEST_LINK_LIBRARIES}"
    )
  else()
    target_link_libraries(
      ${fq_build_target_name}
      PRIVATE
        ${link_object_files}
        ${TEST_LINK_LIBRARIES}
    )
  endif()

  add_dependencies(
    ${fq_build_target_name}
    ${fq_deps_list}
  )

  # 5. Add test runner target
  _add_libc_test_runner(
    ${fq_target_name}
    ${fq_build_target_name}
    "${TEST_ARGS}"
    "${TEST_LOADER_ARGS}"
    "${TEST_ENV}"
  )

  # 6. Register with test suite
  _register_libc_test_with_suite(
    ${fq_target_name}
    ${fq_build_target_name}
    "${TEST_SUITE}"
    "${TEST_BUILD_ONLY}"
  )
endfunction()


# Rule to add a fuzzer test.
# Usage
#    add_libc_fuzzer(
#      <target name>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#    )
function(add_libc_fuzzer target_name)
  llvm_libc_parse_arguments(LIBC_FUZZER ${ARGN})
  if(NOT LIBC_FUZZER_SRCS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a SRCS list of .cpp "
      "files.")
  endif()
  if(NOT LIBC_FUZZER_DEPENDS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a DEPENDS list of "
      "'add_entrypoint_object' targets.")
  endif()

  set(link_libraries ${LIBC_FUZZER_LINK_LIBRARIES})
  if(LIBC_FUZZER_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message(VERBOSE "Fuzz test ${target_name} will be skipped as MPFR library is not available.")
      return()
    endif()
    list(APPEND link_libraries mpfr gmp)
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
    ${link_libraries}
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

endfunction()


# A unified function to add a libc test.
# FULL_BUILD_ONLY: skip test in overlay mode.
# OVERLAY_ONLY: skip test in full build mode.
# FP_TEST: add floating-point test helper dependencies.
# INTEGRATION_TEST: link against IntegrationTest instead of UnitTest framework.
# NEED_MPFR / NEED_MPFR_F80 / NEED_MPFR_F128: link against MPFR; tests requiring MPFR run only in overlay mode.
# NEED_MPC: link against MPC and MPFR; tests requiring MPC run only in overlay mode.
function(add_libc_test test_name)
  add_target_with_flags(
    ${test_name}
    CREATE_TARGET create_libc_test
    ${ARGN}
  )
endfunction()

function(add_fp_unittest test_name)
  add_libc_test(
    ${test_name}
    FP_TEST
    ${ARGN}
  )
endfunction()

function(add_integration_test test_name)
  add_libc_test(
    ${test_name}
    INTEGRATION_TEST
    FULL_BUILD_ONLY
    ${ARGN}
  )
endfunction()

# Tests all implementations that can run on the target CPU.
function(add_libc_multi_impl_test name suite)
  get_property(fq_implementations GLOBAL PROPERTY ${name}_implementations)
  foreach(fq_config_name IN LISTS fq_implementations)
    get_target_property(required_cpu_features ${fq_config_name} REQUIRE_CPU_FEATURES)
    cpu_supports(can_run "${required_cpu_features}")
    if(can_run)
      string(FIND ${fq_config_name} "." last_dot_loc REVERSE)
      math(EXPR name_loc "${last_dot_loc} + 1")
      string(SUBSTRING ${fq_config_name} ${name_loc} -1 target_name)
      add_libc_test(
        ${target_name}_test
        SUITE
          ${suite}
        COMPILE_OPTIONS
          ${LIBC_COMPILE_OPTIONS_NATIVE}
        ${ARGN}
        DEPENDS
          ${fq_config_name}
          libc.src.__support.macros.sanitizer
          libc.test.UnitTest.MemoryMatcher
      )
      get_fq_target_name(${fq_config_name}_test fq_target_name)
    else()
      message(STATUS "Skipping test for '${fq_config_name}' insufficient host cpu features '${required_cpu_features}'")
    endif()
  endforeach()
endfunction()
