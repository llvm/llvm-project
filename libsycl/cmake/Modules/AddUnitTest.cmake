function(add_sycl_unittest test_dirname)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)

  add_unittest(LibsyclUnitTests ${test_dirname}
              $<TARGET_OBJECTS:${LIBSYCL_OBJ_TARGET}> ${ARGN})
  target_compile_definitions(${test_dirname}
                              PRIVATE _LIBSYCL_BUILDING_LIBRARY)

    add_custom_target(check-sycl-${test_dirname}
        ${CMAKE_COMMAND} -E env
        ${CMAKE_CURRENT_BINARY_DIR}/${test_dirname}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${test_dirname}
    )

  add_dependencies(check-sycl-unittests check-sycl-${test_dirname})

  target_link_libraries(${test_dirname}
    PRIVATE
      ol_mock
      # required for fake device images creation
      LLVMFrontendOffloading
      LLVMObject
      LLVMSupport
    )

  target_include_directories(${test_dirname}
    PRIVATE SYSTEM
      ${LIBSYCL_BUILD_INCLUDE_DIR}
      ${LIBSYCL_SOURCE_DIR}/src/
      ${LIBSYCL_SOURCE_DIR}/unittests/
      $<TARGET_PROPERTY:LLVMOffload,INTERFACE_INCLUDE_DIRECTORIES>
      ${LLVM_MAIN_INCLUDE_DIR}
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
