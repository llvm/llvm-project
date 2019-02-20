# Test target to run Python test suite from main build.

add_custom_target(check-clang-python
    COMMAND ${CMAKE_COMMAND} -E env
            CLANG_LIBRARY_PATH=$<TARGET_FILE_DIR:libclang>
            ${PYTHON_EXECUTABLE} -m unittest discover
    DEPENDS libclang
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(RUN_PYTHON_TESTS TRUE)
set_target_properties(check-clang-python PROPERTIES FOLDER "Clang tests")

# Do not try to run if libclang was built with ASan because
# the sanitizer library will likely be loaded too late to perform
# interception and will then fail.
# We could use LD_PRELOAD/DYLD_INSERT_LIBRARIES but this isn't
# portable so its easier just to not run the tests when building
# with ASan.
list(FIND LLVM_USE_SANITIZER "Address" LLVM_USE_ASAN_INDEX)
if(NOT LLVM_USE_ASAN_INDEX EQUAL -1)
  set(RUN_PYTHON_TESTS FALSE)
endif()

# Tests fail on Windows, and need someone knowledgeable to fix.
# It's not clear whether it's a test or a valid binding problem.
if(WIN32)
  set(RUN_PYTHON_TESTS FALSE)
endif()

# AArch64 and Hexagon have known test failures that need to be
# addressed.
# SystemZ has broken Python/FFI interface:
# https://reviews.llvm.org/D52840#1265716
if(${LLVM_NATIVE_ARCH} MATCHES "^(AArch64|Hexagon|SystemZ)$")
  set(RUN_PYTHON_TESTS FALSE)
endif()

if(RUN_PYTHON_TESTS)
    set_property(GLOBAL APPEND PROPERTY
                 LLVM_ADDITIONAL_TEST_TARGETS check-clang-python)
endif()
