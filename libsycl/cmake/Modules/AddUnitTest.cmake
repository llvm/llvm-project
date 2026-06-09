function(add_sycl_unittest test_name)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)

  add_unittest(LibsyclUnitTests ${test_name}
              $<TARGET_OBJECTS:${LIBSYCL_OBJ_TARGET}> ${ARGN})
  target_compile_definitions(${test_name}
                              PRIVATE _LIBSYCL_BUILDING_LIBRARY)

    add_custom_target(check-sycl-${test_name}
        ${CMAKE_COMMAND} -E env
        ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${test_name}
    )

  add_dependencies(check-sycl-unittests check-sycl-${test_name})

  target_link_libraries(${test_name}
    PRIVATE
      ol_mock
      # required for fake device images creation
      LLVMFrontendOffloading
      LLVMObject
      LLVMSupport
    )

  target_include_directories(${test_name}
    PRIVATE SYSTEM
      ${LIBSYCL_BUILD_INCLUDE_DIR}
      ${LIBSYCL_SOURCE_DIR}/src/
      ${LIBSYCL_SOURCE_DIR}/unittests/
      $<TARGET_PROPERTY:LLVMOffload,INTERFACE_INCLUDE_DIRECTORIES>
      ${LLVM_MAIN_INCLUDE_DIR}
    )
  if (UNIX)
    # These warnings are coming from Google Test code.
    target_compile_options(${test_name}
      PRIVATE
        -Wno-unused-parameter
        -Wno-inconsistent-missing-override
    )
  endif()
endfunction()
