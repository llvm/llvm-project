add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
  CrashHandlerFixture.cpp
  Numeric.cpp
  NumericalFormatTest.cpp
  Reduction.cpp
  RuntimeCrashTest.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
