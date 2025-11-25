cmake_minimum_required(VERSION 3.20)
project(DSMILClient VERSION 2.0.1 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler specific options for high performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
endif()

# Build options
option(DSMIL_BUILD_EXAMPLES "Build example applications" ON)
option(DSMIL_BUILD_TESTS "Build test suite" ON)
option(DSMIL_ENABLE_LOGGING "Enable detailed logging" ON)
option(DSMIL_ENABLE_METRICS "Enable performance metrics collection" ON)
option(DSMIL_ENABLE_HARDWARE_SECURITY "Enable hardware security integration" ON)

# Find required dependencies
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find OpenSSL
find_package(OpenSSL REQUIRED)
if(OPENSSL_FOUND)
    message(STATUS "Found OpenSSL ${OPENSSL_VERSION}")
endif()

# Find libcurl
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "Found libcurl ${CURL_VERSION_STRING}")
endif()

# Find optional dependencies
find_package(Boost COMPONENTS system thread QUIET)
if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION} - enabling advanced features")
    set(DSMIL_HAS_BOOST ON)
endif()

# Check for io_uring support (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(URING_INCLUDE_DIR liburing.h)
    find_library(URING_LIBRARY uring)
    if(URING_INCLUDE_DIR AND URING_LIBRARY)
        message(STATUS "Found liburing - enabling high-performance async I/O")
        set(DSMIL_HAS_URING ON)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dsmil/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/dsmil/version.hpp"
    @ONLY
)

# Core library sources
set(DSMIL_CORE_SOURCES
    src/client.cpp
    src/connection_pool.cpp
    src/websocket_client.cpp
    src/bulk_operator.cpp
    src/security_manager.cpp
    src/device_registry.cpp
    src/performance_monitor.cpp
    src/error_handler.cpp
    src/config_manager.cpp
    src/kernel_interface.cpp
    src/async_executor.cpp
    src/thread_pool.cpp
    src/metrics_collector.cpp
    src/audit_logger.cpp
    src/safety_validator.cpp
)

# Hardware security sources (optional)
if(DSMIL_ENABLE_HARDWARE_SECURITY)
    list(APPEND DSMIL_CORE_SOURCES
        src/hardware_security.cpp
        src/tpm_interface.cpp
        src/hardware_crypto.cpp
    )
endif()

# Platform-specific sources
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND DSMIL_CORE_SOURCES
        src/linux/kernel_module_interface.cpp
        src/linux/sysfs_interface.cpp
    )
    if(DSMIL_HAS_URING)
        list(APPEND DSMIL_CORE_SOURCES src/linux/uring_interface.cpp)
    endif()
endif()

# Core library headers
set(DSMIL_CORE_HEADERS
    include/dsmil/client.hpp
    include/dsmil/connection_pool.hpp
    include/dsmil/websocket_client.hpp
    include/dsmil/bulk_operator.hpp
    include/dsmil/security_manager.hpp
    include/dsmil/device_registry.hpp
    include/dsmil/performance_monitor.hpp
    include/dsmil/error_handler.hpp
    include/dsmil/types.hpp
    include/dsmil/config.hpp
    include/dsmil/constants.hpp
    include/dsmil/exceptions.hpp
    include/dsmil/utils.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/include/dsmil/version.hpp
)

# Create the main library
add_library(dsmil_client SHARED ${DSMIL_CORE_SOURCES})
add_library(dsmil_client_static STATIC ${DSMIL_CORE_SOURCES})

# Set library properties
set_target_properties(dsmil_client PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${DSMIL_CORE_HEADERS}"
)

set_target_properties(dsmil_client_static PROPERTIES
    OUTPUT_NAME dsmil_client
    PUBLIC_HEADER "${DSMIL_CORE_HEADERS}"
)

# Include directories for build and install
target_include_directories(dsmil_client PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(dsmil_client_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link required libraries
target_link_libraries(dsmil_client PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    Threads::Threads
)

target_link_libraries(dsmil_client_static PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    Threads::Threads
)

# Optional dependencies
if(DSMIL_HAS_BOOST)
    target_link_libraries(dsmil_client PRIVATE Boost::system Boost::thread)
    target_link_libraries(dsmil_client_static PRIVATE Boost::system Boost::thread)
    target_compile_definitions(dsmil_client PRIVATE DSMIL_HAS_BOOST)
    target_compile_definitions(dsmil_client_static PRIVATE DSMIL_HAS_BOOST)
endif()

if(DSMIL_HAS_URING)
    target_link_libraries(dsmil_client PRIVATE ${URING_LIBRARY})
    target_link_libraries(dsmil_client_static PRIVATE ${URING_LIBRARY})
    target_include_directories(dsmil_client PRIVATE ${URING_INCLUDE_DIR})
    target_include_directories(dsmil_client_static PRIVATE ${URING_INCLUDE_DIR})
    target_compile_definitions(dsmil_client PRIVATE DSMIL_HAS_URING)
    target_compile_definitions(dsmil_client_static PRIVATE DSMIL_HAS_URING)
endif()

# Compile definitions
if(DSMIL_ENABLE_LOGGING)
    target_compile_definitions(dsmil_client PRIVATE DSMIL_ENABLE_LOGGING)
    target_compile_definitions(dsmil_client_static PRIVATE DSMIL_ENABLE_LOGGING)
endif()

if(DSMIL_ENABLE_METRICS)
    target_compile_definitions(dsmil_client PRIVATE DSMIL_ENABLE_METRICS)
    target_compile_definitions(dsmil_client_static PRIVATE DSMIL_ENABLE_METRICS)
endif()

if(DSMIL_ENABLE_HARDWARE_SECURITY)
    target_compile_definitions(dsmil_client PRIVATE DSMIL_ENABLE_HARDWARE_SECURITY)
    target_compile_definitions(dsmil_client_static PRIVATE DSMIL_ENABLE_HARDWARE_SECURITY)
endif()

# Create alias for modern cmake usage
add_library(DSMILClient::DSMILClient ALIAS dsmil_client)
add_library(DSMILClient::DSMILClientStatic ALIAS dsmil_client_static)

# Build examples
if(DSMIL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(DSMIL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

# Install targets
install(TARGETS dsmil_client dsmil_client_static
    EXPORT DSMILClientTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsmil
)

# Install headers
install(DIRECTORY include/dsmil
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the generated version header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/dsmil/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsmil
)

# Generate and install cmake config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DSMILClientConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DSMILClientConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DSMILClient
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DSMILClientConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DSMILClientConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DSMILClientConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DSMILClient
)

# Export targets
install(EXPORT DSMILClientTargets
    FILE DSMILClientTargets.cmake
    NAMESPACE DSMILClient::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DSMILClient
)

# Documentation
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "dsmil-client")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance C++ client library for DSMIL control system")
set(CPACK_PACKAGE_VENDOR "JRTC1 Educational Development")
set(CPACK_GENERATOR "DEB;RPM;TGZ")

# Debian package configuration
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl3 (>= 3.0.0), libcurl4 (>= 7.68.0)")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")

# RPM package configuration
set(CPACK_RPM_PACKAGE_REQUIRES "openssl-libs >= 3.0.0, libcurl >= 7.68.0")
set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "DSMIL C++ SDK Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${DSMIL_BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${DSMIL_BUILD_TESTS}")
message(STATUS "  Enable Logging: ${DSMIL_ENABLE_LOGGING}")
message(STATUS "  Enable Metrics: ${DSMIL_ENABLE_METRICS}")
message(STATUS "  Hardware Security: ${DSMIL_ENABLE_HARDWARE_SECURITY}")
message(STATUS "  Has Boost: ${DSMIL_HAS_BOOST}")
message(STATUS "  Has io_uring: ${DSMIL_HAS_URING}")
message(STATUS "")