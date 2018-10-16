# Test target to run Python test suite from main build.

add_custom_target(check-clang-python
    COMMAND ${CMAKE_COMMAND} -E env
            CLANG_LIBRARY_PATH=$<TARGET_FILE_DIR:libclang>
            ${PYTHON_EXECUTABLE} -m unittest discover
    DEPENDS libclang
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Check if we are building with ASan
list(FIND LLVM_USE_SANITIZER "Address" LLVM_USE_ASAN_INDEX)
if (LLVM_USE_ASAN_INDEX EQUAL -1)
  set(LLVM_USE_ASAN FALSE)
else()
  set(LLVM_USE_ASAN TRUE)
endif()

# Tests fail on Windows, and need someone knowledgeable to fix.
# It's not clear whether it's a test or a valid binding problem.
#
# Do not try to run if libclang was built with ASan because
# the sanitizer library will likely be loaded too late to perform
# interception and will then fail.
# We could use LD_PRELOAD/DYLD_INSERT_LIBRARIES but this isn't
# portable so its easier just to not run the tests when building
# with ASan.
if((NOT WIN32) AND (NOT LLVM_USE_ASAN))
    set_property(GLOBAL APPEND PROPERTY
                 LLVM_ADDITIONAL_TEST_TARGETS check-clang-python)
endif()
