include(AddUnitTest)

add_custom_target(LibsyclUnitTests)

add_custom_target(check-sycl-unittests)

add_subdirectory(mock)
add_subdirectory(device_selector)
add_subdirectory(platform)
add_subdirectory(program_manager)
add_subdirectory(queue)
