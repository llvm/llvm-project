function(add_sycl_unittest_internal test_dirname link_variant is_preview)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)

  get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

  string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)

  # Select which sycl libraries and object to link based
  # on whether this is a preview build.
  if (MSVC AND build_type_lower MATCHES "debug")
    if (${is_preview})
      set(sycl_obj_target "sycl-previewd_object")
      set(sycl_so_target "sycl-previewd")
    else()
      set(sycl_obj_target "sycld_object")
      set(sycl_so_target "sycld")
    endif()
  else()
    if (${is_preview})
      set(sycl_obj_target "sycl-preview_object")
      set(sycl_so_target "sycl-preview")
    else()
      set(sycl_obj_target "sycl_object")
      set(sycl_so_target "sycl")
    endif()
  endif()

  # This is done to ensure that preview tests are kept in a separate
  # directory, so that they do not interfere with the non-preview tests.
  # Chaning CMAKE_CURRENT_BINARY_DIR should not affect this variable in its
  # parent scope.
  if (${is_preview})
      set(CMAKE_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/Preview")
  endif()

  if ("${link_variant}" MATCHES "SHARED")
    set(SYCL_LINK_LIBS ${sycl_so_target})
    add_unittest(SYCLUnitTests ${test_dirname} ${ARGN})
  else()
    add_unittest(SYCLUnitTests ${test_dirname}
                $<TARGET_OBJECTS:${sycl_obj_target}> ${ARGN})
    target_compile_definitions(${test_dirname}
                               PRIVATE __SYCL_BUILD_SYCL_DLL)

    get_target_property(SYCL_LINK_LIBS ${sycl_so_target} LINK_LIBRARIES)
  endif()

  if (SYCL_ENABLE_COVERAGE)
    target_compile_options(${test_dirname} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
    target_link_options(${test_dirname} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
  endif()

  # Add preview-specific compile definition
  if (${is_preview})
    target_compile_definitions(${test_dirname}
        PRIVATE __INTEL_PREVIEW_BREAKING_CHANGES)
    set(sycl_cache_suffix "_preview")
  endif()

  if (SYCL_ENABLE_XPTI_TRACING)
    target_compile_definitions(${test_dirname}
      PRIVATE XPTI_ENABLE_INSTRUMENTATION XPTI_STATIC_LIBRARY)
  endif()

  # check-sycl-unittests was using an old sycl library. So, to get
  # around this problem, we add the new sycl library to the PATH and
  # LD_LIBRARY_PATH on Windows and Linux respectively.
    add_custom_target(check-sycl-${test_dirname}
        ${CMAKE_COMMAND} -E env
        LLVM_PROFILE_FILE="${SYCL_COVERAGE_PATH}/${test_dirname}.profraw"
        SYCL_CONFIG_FILE_NAME=null.cfg
        SYCL_DEVICELIB_NO_FALLBACK=1
        SYCL_CACHE_DIR="${CMAKE_BINARY_DIR}/sycl_cache${sycl_cache_suffix}"
        "LD_LIBRARY_PATH=${SYCL_BINARY_DIR}/unittests/lib:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        ${CMAKE_CURRENT_BINARY_DIR}/${test_dirname}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${test_dirname}
    )

  add_dependencies(check-sycl-unittests check-sycl-${test_dirname})

  target_link_libraries(${test_dirname}
    PRIVATE
      mockOpenCL
      LLVMTestingSupport
      OpenCL-Headers
      emhash::emhash
      unified-runtime::mock
      ${SYCL_LINK_LIBS}
    )

  add_dependencies(${test_dirname} ur_adapter_mock mockOpenCL)

  if(SYCL_ENABLE_EXTENSION_JIT)
    target_link_libraries(${test_dirname} PRIVATE sycl-jit)
  endif(SYCL_ENABLE_EXTENSION_JIT)

  target_include_directories(${test_dirname}
    PRIVATE SYSTEM
      ${sycl_inc_dir}
      ${SYCL_SOURCE_DIR}/source/
      ${SYCL_SOURCE_DIR}/unittests/
    )
  if (UNIX)
    # These warnings are coming from Google Test code.
    target_compile_options(${test_dirname}
      PRIVATE
        -Wno-unused-parameter
        -Wno-inconsistent-missing-override
    )
  endif()
  target_compile_definitions(${test_dirname} PRIVATE SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)
endfunction()

# add_sycl_unittest(test_name_prefix SHARED|OBJECT file1.cpp, file2.cpp ...)
#
# Will compile the list of files together to create two builds, with and without
# the SYCL preview features enabled.
# Produces two binaries, named `basename(test_name_prefix_Non_Preview_Tests)` and `basename(test_name_prefix_Preview_Tests)`
# Note: in case of changing test names below, please also adjust the test suffix
# in sycl/test/Unit/lit.cfg.py in the line which looks like this:
#   `config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, <...>)`
macro(add_sycl_unittest test_name_prefix link_variant)
  add_sycl_unittest_internal(${test_name_prefix}_Non_Preview_Tests ${link_variant} FALSE ${ARGN})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_sycl_unittest_internal(${test_name_prefix}_Preview_Tests ${link_variant} TRUE ${ARGN})
  endif()
endmacro()

add_sycl_unittest(AllowListTests OBJECT 
    ParseAllowList.cpp
    DeviceIsAllowed.cpp
)