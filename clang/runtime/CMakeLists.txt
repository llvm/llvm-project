# TODO: Set the install directory.

include(ExternalProject)

set(known_subdirs
  "libcxx"
  )

foreach (dir ${known_subdirs})
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  endif()
endforeach()

function(get_ext_project_build_command out_var target)
  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(${out_var} "$(MAKE)" "${target}" PARENT_SCOPE)
  else()
    set(${out_var} ${CMAKE_COMMAND} --build . --target ${target}
                                    --config $<CONFIGURATION> PARENT_SCOPE)
  endif()
endfunction()

set(COMPILER_RT_SRC_ROOT ${LLVM_MAIN_SRC_DIR}/projects/compiler-rt)
if(LLVM_BUILD_EXTERNAL_COMPILER_RT AND EXISTS ${COMPILER_RT_SRC_ROOT}/)
  if(CMAKE_VERSION VERSION_GREATER 3.3.20150708)
    set(cmake_3_4_USES_TERMINAL_OPTIONS
      USES_TERMINAL_CONFIGURE 1
      USES_TERMINAL_BUILD 1
      USES_TERMINAL_INSTALL 1
      )
  endif()

  # Add compiler-rt as an external project.
  set(COMPILER_RT_PREFIX ${CMAKE_BINARY_DIR}/projects/compiler-rt)

  set(STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/compiler-rt-stamps/)
  set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/compiler-rt-bins/)

  add_custom_target(compiler-rt-clear
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${STAMP_DIR}
    COMMENT "Clobberring compiler-rt build and stamp directories"
    )

  # Find all variables that start with COMPILER_RT and populate a variable with
  # them.
  get_cmake_property(variableNames VARIABLES)
  foreach(varaibleName ${variableNames})
    if(varaibleName MATCHES "^COMPILER_RT")
      string(REPLACE ";" "\;" value "${${varaibleName}}")
      list(APPEND COMPILER_RT_PASSTHROUGH_VARIABLES
        -D${varaibleName}=${${value}})
    endif()
  endforeach()

  ExternalProject_Add(compiler-rt
    PREFIX ${COMPILER_RT_PREFIX}
    SOURCE_DIR ${COMPILER_RT_SRC_ROOT}
    STAMP_DIR ${STAMP_DIR}
    BINARY_DIR ${BINARY_DIR}
    CMAKE_ARGS ${CLANG_COMPILER_RT_CMAKE_ARGS}
               -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
               -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DLLVM_CONFIG_PATH=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-config
               -DCOMPILER_RT_OUTPUT_DIR=${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}
               -DCOMPILER_RT_EXEC_OUTPUT_DIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
               -DCOMPILER_RT_INSTALL_PATH:STRING=lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}
               -DCOMPILER_RT_INCLUDE_TESTS=${LLVM_INCLUDE_TESTS}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               ${COMPILER_RT_PASSTHROUGH_VARIABLES}
    INSTALL_COMMAND ""
    STEP_TARGETS configure build
    ${cmake_3_4_USES_TERMINAL_OPTIONS}
    )

  get_ext_project_build_command(run_clean_compiler_rt clean)
  ExternalProject_Add_Step(compiler-rt clean
    COMMAND ${run_clean_compiler_rt}
    COMMENT "Cleaning compiler-rt..."
    DEPENDEES configure
    DEPENDERS build
    DEPENDS clang
    WORKING_DIRECTORY ${BINARY_DIR}
    )

  add_dependencies(compiler-rt-configure clang llvm-config)

  install(CODE "execute_process\(COMMAND \${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX} -P ${BINARY_DIR}/cmake_install.cmake \)"
    COMPONENT compiler-rt)

  add_custom_target(install-compiler-rt
                    DEPENDS compiler-rt
                    COMMAND "${CMAKE_COMMAND}"
                             -DCMAKE_INSTALL_COMPONENT=compiler-rt
                             -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")

  # Add top-level targets that build specific compiler-rt runtimes.
  set(COMPILER_RT_RUNTIMES asan builtins dfsan lsan msan profile tsan ubsan)
  foreach(runtime ${COMPILER_RT_RUNTIMES})
    get_ext_project_build_command(build_runtime_cmd ${runtime})
    add_custom_target(${runtime}
      COMMAND ${build_runtime_cmd}
      DEPENDS compiler-rt-configure
      WORKING_DIRECTORY ${BINARY_DIR}
      VERBATIM)
  endforeach()

  if(LLVM_INCLUDE_TESTS)
    # Add binaries that compiler-rt tests depend on.
    set(COMPILER_RT_TEST_DEPENDENCIES
      FileCheck count not llvm-nm llvm-symbolizer)

    # Add top-level targets for various compiler-rt test suites.
    set(COMPILER_RT_TEST_SUITES check-asan check-asan-dynamic check-dfsan
      check-lsan check-msan check-sanitizer check-tsan check-ubsan)
    foreach(test_suite ${COMPILER_RT_TEST_SUITES})
      get_ext_project_build_command(run_test_suite ${test_suite})
      add_custom_target(${test_suite}
        COMMAND ${run_test_suite}
        DEPENDS compiler-rt-build ${COMPILER_RT_TEST_DEPENDENCIES}
        WORKING_DIRECTORY ${BINARY_DIR}
        VERBATIM)
    endforeach()

    # Add special target to run all compiler-rt test suites.
    get_ext_project_build_command(run_check_compiler_rt check-all)
    add_custom_target(check-compiler-rt
      COMMAND ${run_check_compiler_rt}
      DEPENDS compiler-rt-build ${COMPILER_RT_TEST_DEPENDENCIES}
      WORKING_DIRECTORY ${BINARY_DIR}
      VERBATIM)
  endif()
endif()
