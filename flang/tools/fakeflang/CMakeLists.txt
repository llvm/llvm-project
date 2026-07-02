set(LLVM_LINK_COMPONENTS
  Support
)
set(FLANG_BUILD_TOOLS OFF) # Do not install
add_flang_tool(fakeflang
  fakeflang.cpp
)
target_compile_definitions(fakeflang PRIVATE "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")

# Using $<TARGET_FILE:file> in an add_custom_command would automatically
# add a dependency.
set(_flang_exe "${LLVM_RUNTIME_OUTPUT_INTDIR}/flang${CMAKE_EXECUTABLE_SUFFIX}")

# Create ${_flang_exe} using fakeflang if it does not exist yet.
# Cannot use an OUTPUT/BYPRODUCT as it would make ninja complain about multiple
# rules to create ${_flang_exe} (the other being being the full flang).
# Automatically execute it after fakeflang has been built instead.
add_custom_command(POST_BUILD
  TARGET fakeflang
  COMMAND "${CMAKE_COMMAND}" "-DINPUT_FILE=$<TARGET_FILE:fakeflang>" "-DOUTPUT_FILE=${_flang_exe}" "-P" "${CMAKE_CURRENT_SOURCE_DIR}/ensure_flang_exists.cmake"
)

# Phony target that ensures bin/flang exists; if the full flang has not been
# compiled yet, use a mock flang driver only for passing CMake's
# CMake_Fortran_COMPILER tests.
add_custom_target(flang-lazy)
add_dependencies(flang-lazy fakeflang)
