if(CMAKE_CLC_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_CLC_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# Remove any cached result from an older CMake version.
# We now store this in CMakeCCompiler.cmake.
unset(CMAKE_CLC_COMPILER_WORKS CACHE)

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that that selected CLC compiler can actually compile
# and link the most basic of programs. If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_CLC_COMPILER_WORKS)
  PrintTestCompilerStatus("CLC" "")
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCLCCompiler.cl
    "__kernel void test_k(global int * a)\n"
    "{ *a = 1; }\n")
  try_compile(CMAKE_CLC_COMPILER_WORKS ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCLCCompiler.cl
    # We never generate executable so bypass the link step
    CMAKE_FLAGS -DCMAKE_CLC_LINK_EXECUTABLE='true'
    OUTPUT_VARIABLE __CMAKE_CLC_COMPILER_OUTPUT)
  # Move result from cache to normal variable.
  set(CMAKE_CLC_COMPILER_WORKS ${CMAKE_CLC_COMPILER_WORKS})
  unset(CMAKE_CLC_COMPILER_WORKS CACHE)
  set(CLC_TEST_WAS_RUN 1)
endif()

if(NOT CMAKE_CLC_COMPILER_WORKS)
  PrintTestCompilerStatus("CLC" " -- broken")
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining if the CLC compiler works failed with "
    "the following output:\n${__CMAKE_CLC_COMPILER_OUTPUT}\n\n")
  message(FATAL_ERROR "The CLC compiler \"${CMAKE_CLC_COMPILER}\" "
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n ${__CMAKE_CLC_COMPILER_OUTPUT}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(CLC_TEST_WAS_RUN)
    PrintTestCompilerStatus("CLC" " -- works")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the CLC compiler works passed with "
      "the following output:\n${__CMAKE_CLC_COMPILER_OUTPUT}\n\n")
  endif()

  include(${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake)

endif()

unset(__CMAKE_CLC_COMPILER_OUTPUT)
