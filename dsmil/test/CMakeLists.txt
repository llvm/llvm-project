# Test build configuration for DSLLVM

cmake_minimum_required(VERSION 3.15)

# Runtime tests
add_executable(test_ot_telemetry_runtime
    runtime/test_ot_telemetry.c
)
target_link_libraries(test_ot_telemetry_runtime
    dsmil_ot_telemetry
)
target_include_directories(test_ot_telemetry_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_fuzz_telemetry_runtime
    runtime/test_fuzz_telemetry.c
)
target_link_libraries(test_fuzz_telemetry_runtime
    dsmil_fuzz_telemetry
)
target_include_directories(test_fuzz_telemetry_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_fuzz_telemetry_advanced_runtime
    runtime/test_fuzz_telemetry_advanced.c
)
target_link_libraries(test_fuzz_telemetry_advanced_runtime
    dsmil_fuzz_telemetry_advanced
)
target_include_directories(test_fuzz_telemetry_advanced_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Integration tests
add_executable(test_telecom_macros
    integration/test_telecom_macros.c
)
target_link_libraries(test_telecom_macros
    dsmil_ot_telemetry
)
target_include_directories(test_telecom_macros PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_attributes
    integration/test_attributes.c
)
target_link_libraries(test_attributes
    dsmil_ot_telemetry
    dsmil_fuzz_telemetry
)
target_include_directories(test_attributes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Pass tests (compile-only, use LIT for execution)
add_library(test_ot_telemetry_pass OBJECT
    passes/test_ot_telemetry_pass.c
)
target_include_directories(test_ot_telemetry_pass PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_library(test_telecom_pass OBJECT
    passes/test_telecom_pass.c
)
target_include_directories(test_telecom_pass PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_library(test_fuzz_coverage_pass OBJECT
    passes/test_fuzz_coverage_pass.c
)
target_include_directories(test_fuzz_coverage_pass PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Test targets
add_custom_target(check-dsmil-runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_ot_telemetry_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_fuzz_telemetry_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_fuzz_telemetry_advanced_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_telecom_macros
    COMMAND ${CMAKE_BINARY_DIR}/test_attributes
    DEPENDS
        test_ot_telemetry_runtime
        test_fuzz_telemetry_runtime
        test_fuzz_telemetry_advanced_runtime
        test_telecom_macros
        test_attributes
    COMMENT "Running DSLLVM runtime tests"
)

# Coverage target
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    
    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${GENHTML_PATH} coverage.info --output-directory coverage
            COMMENT "Generating coverage report"
        )
    endif()
endif()
