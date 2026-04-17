function(add_sycl_unittest test_dirname)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)

  get_target_property(LIBSYCL_BINARY_DIR libsycl-toolchain BINARY_DIR)

  string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)

  set(sycl_obj_target "sycl_object")

  add_unittest(LibsyclUnitTests ${test_dirname}
              $<TARGET_OBJECTS:${sycl_obj_target}> ${ARGN})
  target_compile_definitions(${test_dirname}
                              PRIVATE _LIBSYCL_BUILDING_LIBRARY)

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
      ${UT_LIB_TARGET_NAME}
      ol_mock
    )

  target_include_directories(${test_dirname}
    PRIVATE SYSTEM
      ${sycl_inc_dir}
      ${SYCL_SOURCE_DIR}/src/
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
endfunction()

add_sycl_unittest(USMTests 
    usm_alloc.cpp
)