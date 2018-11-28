if(CMAKE_LLAsm_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_LLAsm_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# Remove any cached result from an older CMake version.
# We now store this in CMakeCCompiler.cmake.
unset(CMAKE_LLAsm_COMPILER_WORKS CACHE)

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that that selected llvm assembler can actually compile
# and link the most basic of programs. If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_LLAsm_COMPILER_WORKS)
  PrintTestCompilerStatus("LLAsm" "")
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testLLAsmCompiler.ll
    "define i32 @test() {\n"
    "ret i32 0 }\n" )
  try_compile(CMAKE_LLAsm_COMPILER_WORKS ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testLLAsmCompiler.ll
    # We never generate executable so bypass the link step
    CMAKE_FLAGS -DCMAKE_LLAsm_LINK_EXECUTABLE='true'
    OUTPUT_VARIABLE __CMAKE_LLAsm_COMPILER_OUTPUT)
  # Move result from cache to normal variable.
  set(CMAKE_LLAsm_COMPILER_WORKS ${CMAKE_LLAsm_COMPILER_WORKS})
  unset(CMAKE_LLAsm_COMPILER_WORKS CACHE)
  set(LLAsm_TEST_WAS_RUN 1)
endif()

if(NOT CMAKE_LLAsm_COMPILER_WORKS)
  PrintTestCompilerStatus("LLAsm" " -- broken")
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining if the LLAsm compiler works failed with "
    "the following output:\n${__CMAKE_LLAsm_COMPILER_OUTPUT}\n\n")
  message(FATAL_ERROR "The LLAsm compiler \"${CMAKE_LLAsm_COMPILER}\" "
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n ${__CMAKE_LLAsm_COMPILER_OUTPUT}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(LLAsm_TEST_WAS_RUN)
    PrintTestCompilerStatus("LLAsm" " -- works")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the LLAsm compiler works passed with "
      "the following output:\n${__CMAKE_LLAsm_COMPILER_OUTPUT}\n\n")
  endif()

  include(${CMAKE_PLATFORM_INFO_DIR}/CMakeLLAsmCompiler.cmake)

endif()

unset(__CMAKE_LLAsm_COMPILER_OUTPUT)
