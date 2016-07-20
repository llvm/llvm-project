# Allow the source files to find headers in src/
include_directories(${PROJECT_SOURCE_DIR}/src)

# Define the source files
set(SOURCE_FILES "benchmark.cc" "colorprint.cc" "commandlineflags.cc"
                 "console_reporter.cc" "csv_reporter.cc" "json_reporter.cc"
                 "log.cc" "reporter.cc" "sleep.cc" "string_util.cc"
                 "sysinfo.cc" "walltime.cc" "complexity.cc")
# Determine the correct regular expression engine to use
if(HAVE_STD_REGEX)
  set(RE_FILES "re_std.cc")
elseif(HAVE_GNU_POSIX_REGEX)
  set(RE_FILES "re_posix.cc")
elseif(HAVE_POSIX_REGEX)
  set(RE_FILES "re_posix.cc")
else()
  message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
endif()

add_library(benchmark ${SOURCE_FILES} ${RE_FILES})


set_target_properties(benchmark PROPERTIES
  OUTPUT_NAME "benchmark"
  VERSION ${GENERIC_LIB_VERSION}
  SOVERSION ${GENERIC_LIB_SOVERSION}
)

# Link threads.
target_link_libraries(benchmark ${CMAKE_THREAD_LIBS_INIT})

# We need extra libraries on Windows
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  target_link_libraries(benchmark Shlwapi)
endif()

# Expose public API
target_include_directories(benchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Install target (will install the library to specified CMAKE_INSTALL_PREFIX variable)
install(
  TARGETS benchmark
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  COMPONENT library)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/benchmark"
  DESTINATION include
  FILES_MATCHING PATTERN "*.*h")
