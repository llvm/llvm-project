# Infrastructure to build flang driver entry point. Flang driver depends on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Option
  Support
  TargetParser
)

set(FLANG_BOLT_ALLOWLIST INSTRUMENT PERF LBR)
set(FLANG_BOLT OFF CACHE STRING "Apply BOLT optimization to flang. \
May be specified as one of ${FLANG_BOLT_ALLOWLIST} to use a particular profiling \
  mechanism.")
set_property(CACHE FLANG_BOLT PROPERTY STRINGS OFF ${FLANG_BOLT_ALLOWLIST})
string(TOUPPER "${FLANG_BOLT}" FLANG_BOLT)
if (FLANG_BOLT AND NOT FLANG_BOLT IN_LIST FLANG_BOLT_ALLOWLIST)
    message(FATAL_ERROR "Specified FLANG_BOLT value '${FLANG_BOLT}' is not one of ${FLANG_BOLT_ALLOWLIST}.")
endif()

if (FLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  set(FLANG_BOLT_DEPS clear-flang-bolt-fdata llvm-bolt llvm-readobj clang-bolt)
  if (NOT FLANG_BOLT STREQUAL "INSTRUMENT")
    list(APPEND FLANG_BOLT_DEPS clear-flang-perf-data)
  endif()
endif()

add_flang_tool(flang
  driver.cpp
  fc1_main.cpp

  DEPENDS
  ${FLANG_BOLT_DEPS}
)

target_link_libraries(flang
  PRIVATE
  flangFrontend
  flangFrontendTool
)

clang_target_link_libraries(flang
  PRIVATE
  clangDriver
  clangOptions
  clangBasic
)

# This creates the executable with a version appended
# and creates a symlink to it without the version
if(CYGWIN OR NOT WIN32) # but it doesn't work on Windows
  set_target_properties(flang PROPERTIES VERSION ${FLANG_EXECUTABLE_VERSION})
endif()

option(FLANG_PLUGIN_SUPPORT "Build Flang with plugin support." ON)

# Enable support for plugins, which need access to symbols from flang
if(FLANG_PLUGIN_SUPPORT)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_DIR})
  export_executable_symbols_for_plugins(flang)
endif()

install(TARGETS flang DESTINATION "${CMAKE_INSTALL_BINDIR}")

if (FLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  # Add a flang-bolt target for backwards compatibility.
  add_custom_target(flang-bolt DEPENDS flang)

  set(FLANG_BOLT_INSTRUMENTED "flang-bolt.inst" CACHE STRING
    "Name of BOLT-instrumented flang binary")
  set(FLANG_INSTRUMENTED "${LLVM_RUNTIME_OUTPUT_INTDIR}/${FLANG_BOLT_INSTRUMENTED}")
  set(PERF_TRAINING_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../utils/perf-training")
  set(FLANG_BOLT_FDATA "${PERF_TRAINING_BINARY_DIR}/flang-prof.fdata")
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
  )
  set(LIT_COMMAND "${lit_base_dir}/${lit_file_name}")

  set(FLANG_BOLT_INPUTS "$<TARGET_FILE:flang>")
  set(FLANG_INSTRUMENTED_OUTPUTS "${FLANG_INSTRUMENTED}")

  # Add in dynamically linked libraries, if needs be. Currently only supported
  # on Linux because it relies on LD_PRELOAD for instrumentation.
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (LLVM_LINK_LLVM_DYLIB)
      set(LLVM_BOLT_INSTRUMENTED "LLVM-bolt.inst" CACHE STRING
        "Name of BOLT-instrumented LLVM library")
      set(LLVM_INSTRUMENTED "${LLVM_RUNTIME_OUTPUT_INTDIR}/${LLVM_BOLT_INSTRUMENTED}")
      list(APPEND FLANG_BOLT_INPUTS "$<TARGET_FILE:LLVM>")
      list(APPEND FLANG_INSTRUMENTED_OUTPUTS "${LLVM_INSTRUMENTED}")
    endif()
  endif()

  # This POST_BUILD command is executed unconditionally even if the flang target
  # is already built.  We need to wrap the whole bolt optimization process in
  # a single python wrapper, so that we can first check if the binary has
  # already been optimized and then exit early with a 0 status if it has.
  add_custom_command(
    TARGET flang POST_BUILD
    COMMAND  "${Python3_EXECUTABLE}" "${LLVM_MAIN_SRC_DIR}/../clang/utils/perf-training/perf-helper.py"
             bolt-optimize
             --method "${FLANG_BOLT}"
             --input "${FLANG_BOLT_INPUTS}"
             --instrumented-output "${FLANG_INSTRUMENTED_OUTPUTS}"
             --fdata "${FLANG_BOLT_FDATA}"
             --perf-training-binary-dir "${PERF_TRAINING_BINARY_DIR}"
             --readelf "$<TARGET_FILE:llvm-readobj>"
             --bolt "$<TARGET_FILE:llvm-bolt>"
             --lit "${LIT_COMMAND}"
             --merge-fdata "$<TARGET_FILE:merge-fdata>"
    COMMENT "Optimizing flang with BOLT"
    USES_TERMINAL
    VERBATIM
  )
endif()

# Keep "flang-new" as a symlink for backwards compatiblity. Remove once "flang"
# is a widely adopted name.
add_flang_symlink(flang-new flang)
