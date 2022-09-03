set(UNITTEST_SOURCES
  adt_test.cpp
  c_api_test.cpp
  endian_test.cpp
  error_test.cpp
  executor_address_test.cpp
  extensible_rtti_test.cpp
  orc_unit_test_main.cpp
  wrapper_function_utils_test.cpp
  simple_packed_serialization_test.cpp
  )

if (COMPILER_RT_CAN_EXECUTE_TESTS)
  add_orc_unittest(OrcUnitTest SOURCES ${UNITTEST_SOURCES})
endif()
