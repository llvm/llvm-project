add_clang_unittest(libclangTests
  LibclangTest.cpp
  DriverTest.cpp
  LINK_LIBS
  libclang
  )

add_subdirectory(CrashTests)
