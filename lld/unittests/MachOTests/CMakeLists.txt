
add_lld_unittest(lldMachOOldTests
  MachONormalizedFileBinaryReaderTests.cpp
  MachONormalizedFileBinaryWriterTests.cpp
  MachONormalizedFileToAtomsTests.cpp
  MachONormalizedFileYAMLTests.cpp
  )

target_link_libraries(lldMachOOldTests
  PRIVATE
  lldDriver
  lldMachOOld
  lldYAML
  )
