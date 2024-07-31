# Test usage of LLD as a library.
# This test covers a more singular case where only one LLD driver is used in the
# target application executable.

add_lld_unittests(LLDAsLibELFTests
  ROCm.cpp
  SomeDrivers.cpp
)

target_link_libraries(LLDAsLibELFTests
  PRIVATE
  lldCommon
  lldELF
)
