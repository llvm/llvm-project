# FIXME(#137855): Ideally we'd fold this into AllClangUnitTests, but it fails
# for unknown reasons. Until that is fixed, link it as a distinct gtest binary.
add_distinct_clang_unittest(libclangCrashTests
  LibclangCrashTest.cpp
  LINK_LIBS
  libclang
  )
copy_libclang_dll_to_unittest(libclangCrashTests)
