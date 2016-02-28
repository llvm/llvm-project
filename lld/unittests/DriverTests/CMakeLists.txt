add_lld_unittest(DriverTests
  UniversalDriverTest.cpp
  DarwinLdDriverTest.cpp
  )

target_link_libraries(DriverTests
  lldDriver
  lldCOFF
  lldCore
  lldMachO
  )
