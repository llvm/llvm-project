add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
  CrashHandlerFixture.cpp
  Format.cpp
  MiscIntrinsic.cpp
  Namelist.cpp
  Numeric.cpp
  NumericalFormatTest.cpp
  Random.cpp
  Reduction.cpp
  RuntimeCrashTest.cpp
  ListInputTest.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
