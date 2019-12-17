add_clang_unittest(libclangTests
  LibclangTest.cpp
  DriverTest.cpp
  )

target_link_libraries(libclangTests
  PRIVATE
  libclang
  )

add_subdirectory(CrashTests)
