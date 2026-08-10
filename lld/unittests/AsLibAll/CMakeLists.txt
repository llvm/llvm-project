# Test usage of LLD as a library.
# This test covers a typical case where all the LLD drivers are pulled into the
# application executable.

add_lld_unittests(LLDAsLibAllTests
  AllDrivers.cpp
)

target_link_libraries(LLDAsLibAllTests
  PRIVATE
  lldCommon
  lldCOFF
  lldELF
  lldMachO
  lldMinGW
  lldWasm
)
