# LLDB Fortran Support Configuration
# This module handles configuration specific to Fortran language support in LLDB

# Option to enable Fortran support testing
option(LLDB_TEST_FORTRAN "Enable Fortran language tests" ON)

# Find Fortran compiler for testing
if(LLDB_TEST_FORTRAN)
  include(CheckLanguage)
  check_language(Fortran)
  
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    message(STATUS "Found Fortran compiler: ${CMAKE_Fortran_COMPILER}")
    
    # Check if it's gfortran or flang
    execute_process(
      COMMAND ${CMAKE_Fortran_COMPILER} --version
      OUTPUT_VARIABLE FORTRAN_COMPILER_VERSION
      ERROR_QUIET
    )
    
    if(FORTRAN_COMPILER_VERSION MATCHES "GNU Fortran")
      set(LLDB_TEST_FORTRAN_COMPILER "gfortran")
    elseif(FORTRAN_COMPILER_VERSION MATCHES "flang")
      set(LLDB_TEST_FORTRAN_COMPILER "flang")
    else()
      set(LLDB_TEST_FORTRAN_COMPILER "unknown")
    endif()
    
    message(STATUS "Fortran compiler type: ${LLDB_TEST_FORTRAN_COMPILER}")
  else()
    message(WARNING "No Fortran compiler found. Fortran tests will be disabled.")
    set(LLDB_TEST_FORTRAN OFF)
  endif()
endif()

# Export variables for use in test configuration
set(LLDB_TEST_FORTRAN ${LLDB_TEST_FORTRAN} PARENT_SCOPE)
set(LLDB_TEST_FORTRAN_COMPILER ${LLDB_TEST_FORTRAN_COMPILER} PARENT_SCOPE)

# Add Fortran test directory if enabled
if(LLDB_TEST_FORTRAN)
  list(APPEND LLDB_TEST_DEPS lldb-test)
  
  # Add custom target for Fortran tests only
  add_custom_target(check-lldb-lang-fortran
    COMMAND ${CMAKE_COMMAND} -E echo "Running LLDB Fortran language tests..."
    COMMAND ${Python3_EXECUTABLE} ${LLVM_MAIN_SRC_DIR}/llvm/utils/lit/lit.py
      --param lldb_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/lit.site.cfg.py
      ${CMAKE_CURRENT_SOURCE_DIR}/test/API/lang/fortran/
    DEPENDS ${LLDB_TEST_DEPS}
    COMMENT "Running LLDB Fortran language tests"
    USES_TERMINAL
  )
endif()