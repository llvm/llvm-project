# - macro_add_check_test(test_name test_source linklib1 ... linklibN)

ENABLE_TESTING()
include(CTest)
set(CMAKE_C_FLAGS_PROFILING "-g -pg")

macro (MACRO_ADD_CHECK_TEST _testName _testSource)
  add_executable(${_testName} ${_testSource})
  target_link_libraries(${_testName} ${ARGN})
  get_target_property(_targetLocation ${_testName} LOCATION) 
  add_test(${_testName} ${_targetLocation})
endmacro (MACRO_ADD_CHECK_TEST)
