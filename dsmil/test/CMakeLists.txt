# Test build configuration for DSLLVM

cmake_minimum_required(VERSION 3.15)
project(DsmilTests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(DSMIL_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Runtime libraries
add_library(dsmil_ot_telemetry STATIC
    ${DSMIL_ROOT_DIR}/runtime/dsmil_ot_telemetry.c
)
target_include_directories(dsmil_ot_telemetry PUBLIC
    ${DSMIL_ROOT_DIR}/include
)
target_compile_definitions(dsmil_ot_telemetry PUBLIC
    _POSIX_C_SOURCE=200809L
    _GNU_SOURCE
)

add_library(dsmil_fuzz_telemetry STATIC
    ${DSMIL_ROOT_DIR}/runtime/dsmil_fuzz_telemetry.c
)
target_include_directories(dsmil_fuzz_telemetry PUBLIC
    ${DSMIL_ROOT_DIR}/include
)
target_compile_definitions(dsmil_fuzz_telemetry PUBLIC
    _POSIX_C_SOURCE=200809L
    _GNU_SOURCE
)
target_link_libraries(dsmil_fuzz_telemetry PUBLIC pthread)

add_library(dsmil_fuzz_telemetry_advanced STATIC
    ${DSMIL_ROOT_DIR}/runtime/dsmil_fuzz_telemetry_advanced.c
)
target_include_directories(dsmil_fuzz_telemetry_advanced PUBLIC
    ${DSMIL_ROOT_DIR}/include
)
target_compile_definitions(dsmil_fuzz_telemetry_advanced PUBLIC
    _POSIX_C_SOURCE=200809L
    _GNU_SOURCE
)
target_link_libraries(dsmil_fuzz_telemetry_advanced PUBLIC pthread)

# Runtime tests
add_executable(test_ot_telemetry_runtime
    runtime/test_ot_telemetry.c
)
target_link_libraries(test_ot_telemetry_runtime
    dsmil_ot_telemetry
)
target_include_directories(test_ot_telemetry_runtime PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_executable(test_fuzz_telemetry_runtime
    runtime/test_fuzz_telemetry.c
)
target_link_libraries(test_fuzz_telemetry_runtime
    dsmil_fuzz_telemetry
)
target_include_directories(test_fuzz_telemetry_runtime PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_executable(test_fuzz_telemetry_advanced_runtime
    runtime/test_fuzz_telemetry_advanced.c
)
target_link_libraries(test_fuzz_telemetry_advanced_runtime
    dsmil_fuzz_telemetry_advanced
)
target_include_directories(test_fuzz_telemetry_advanced_runtime PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

# Integration tests
add_executable(test_telecom_macros
    integration/test_telecom_macros.c
)
target_link_libraries(test_telecom_macros
    dsmil_ot_telemetry
)
target_include_directories(test_telecom_macros PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_executable(test_attributes
    integration/test_attributes.c
)
target_link_libraries(test_attributes
    dsmil_ot_telemetry
    dsmil_fuzz_telemetry
)
target_include_directories(test_attributes PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_executable(test_telemetry_expansion_runtime
    runtime/test_telemetry_expansion_runtime.c
)
target_link_libraries(test_telemetry_expansion_runtime
    dsmil_ot_telemetry
)
target_include_directories(test_telemetry_expansion_runtime PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_executable(test_telemetry_expansion
    integration/test_telemetry_expansion.c
)
target_link_libraries(test_telemetry_expansion
    dsmil_ot_telemetry
)
target_include_directories(test_telemetry_expansion PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

# Pass tests (compile-only, use LIT for execution)
add_library(test_ot_telemetry_pass OBJECT
    passes/test_ot_telemetry_pass.c
)
target_include_directories(test_ot_telemetry_pass PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_library(test_telecom_pass OBJECT
    passes/test_telecom_pass.c
)
target_include_directories(test_telecom_pass PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_library(test_fuzz_coverage_pass OBJECT
    passes/test_fuzz_coverage_pass.c
)
target_include_directories(test_fuzz_coverage_pass PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

add_library(test_telemetry_expansion_pass OBJECT
    passes/test_telemetry_expansion_pass.c
)
target_include_directories(test_telemetry_expansion_pass PRIVATE
    ${DSMIL_ROOT_DIR}/include
)

# Test targets
add_custom_target(check-dsmil-runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_ot_telemetry_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_fuzz_telemetry_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_fuzz_telemetry_advanced_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_telecom_macros
    COMMAND ${CMAKE_BINARY_DIR}/test_attributes
    COMMAND ${CMAKE_BINARY_DIR}/test_telemetry_expansion_runtime
    COMMAND ${CMAKE_BINARY_DIR}/test_telemetry_expansion
    DEPENDS
        test_ot_telemetry_runtime
        test_fuzz_telemetry_runtime
        test_fuzz_telemetry_advanced_runtime
        test_telecom_macros
        test_attributes
        test_telemetry_expansion_runtime
        test_telemetry_expansion
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
