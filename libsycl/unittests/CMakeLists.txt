include(AddUnitTest)

add_custom_target(LibsyclUnitTests)

add_custom_target(check-sycl-unittests)

add_subdirectory(mock)
add_subdirectory(usm_functions)
add_subdirectory(program_manager)