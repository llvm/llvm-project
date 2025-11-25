cmake_minimum_required(VERSION 3.20)

# Find testing framework
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Download and build GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188aa6336f73e3181ff.zip
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Test sources
set(TEST_SOURCES
    test_client.cpp
    test_connection_pool.cpp
    test_kernel_interface.cpp
    test_device_registry.cpp
    test_bulk_operations.cpp
    test_async_operations.cpp
    test_security.cpp
    test_performance.cpp
    test_error_handling.cpp
    test_configuration.cpp
)

# Create test executable
add_executable(dsmil_tests ${TEST_SOURCES})

target_link_libraries(dsmil_tests 
    DSMILClient::DSMILClient
    gtest_main
    gmock_main
)

# Include directories
target_include_directories(dsmil_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# Test configuration
target_compile_definitions(dsmil_tests PRIVATE
    DSMIL_TEST_MODE=1
    DSMIL_MOCK_KERNEL_MODULE=1
)

# Test discovery
include(GoogleTest)
gtest_discover_tests(dsmil_tests)

# Custom test targets
add_custom_target(test_unit
    COMMAND dsmil_tests --gtest_filter=Unit*
    DEPENDS dsmil_tests
    COMMENT "Running unit tests"
)

add_custom_target(test_integration
    COMMAND dsmil_tests --gtest_filter=Integration*
    DEPENDS dsmil_tests
    COMMENT "Running integration tests"
)

add_custom_target(test_performance
    COMMAND dsmil_tests --gtest_filter=Performance*
    DEPENDS dsmil_tests
    COMMENT "Running performance tests"
)

add_custom_target(test_security
    COMMAND dsmil_tests --gtest_filter=Security*
    DEPENDS dsmil_tests
    COMMENT "Running security tests"
)

# Test data files
configure_file(test_config.json test_config.json COPYONLY)
configure_file(test_certificates/test_cert.pem test_certificates/test_cert.pem COPYONLY)
configure_file(test_certificates/test_key.pem test_certificates/test_key.pem COPYONLY)