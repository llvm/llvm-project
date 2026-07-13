add_flang_unittest(FlangCommonTests
  EnumClassTests.cpp
  FastIntSetTest.cpp
  FortranFeaturesTest.cpp
)
target_link_libraries(FlangCommonTests PRIVATE FortranSupport)
