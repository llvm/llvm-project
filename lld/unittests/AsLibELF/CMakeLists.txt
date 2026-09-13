# Test usage of LLD as a library.
# This test covers a more singular case where only one LLD driver is used in the
# target application executable.

add_unittest_with_input_files(LLDUnitTests LLDAsLibELFTests
  OutputStream.cpp
  ROCm.cpp
  SomeDrivers.cpp
)

target_link_libraries(LLDAsLibELFTests
  PRIVATE
  lldCommon
  lldELF
  LLVMTestingSupport
)
