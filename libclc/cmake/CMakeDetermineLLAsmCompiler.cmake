include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

if(NOT CMAKE_LLAsm_PREPROCESSOR)
  find_program(CMAKE_LLAsm_PREPROCESSOR NAMES clang)
endif()
mark_as_advanced(CMAKE_LLAsm_PREPROCESSOR)

if(NOT CMAKE_LLAsm_COMPILER)
  find_program(CMAKE_LLAsm_COMPILER NAMES llvm-as)
endif()
mark_as_advanced(CMAKE_LLAsm_ASSEMBLER)

if(NOT CMAKE_LLAsm_ARCHIVE)
  find_program(CMAKE_LLAsm_ARCHIVE NAMES llvm-link)
endif()
mark_as_advanced(CMAKE_LLAsm_ARCHIVE)

set(CMAKE_LLAsm_PREPROCESSOR_ENV_VAR "LL_PREPROCESSOR")
set(CMAKE_LLAsm_COMPILER_ENV_VAR "LL_ASSEMBLER")
set(CMAKE_LLAsm_ARCHIVE_ENV_VAR "LL_LINKER")
find_file(ll_comp_in CMakeLLAsmCompiler.cmake.in PATHS ${CMAKE_ROOT}/Modules ${CMAKE_MODULE_PATH})
# configure all variables set in this file
configure_file(${ll_comp_in} ${CMAKE_PLATFORM_INFO_DIR}/CMakeLLAsmCompiler.cmake @ONLY)
mark_as_advanced(ll_comp_in)
