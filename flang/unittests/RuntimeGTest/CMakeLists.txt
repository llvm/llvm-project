add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
  RuntimeCrashTest.cpp
  CrashHandlerFixture.cpp
  NumericalFormatTest.cpp
  RuntimeCrashTest.cpp
  CrashHandlerFixture.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
