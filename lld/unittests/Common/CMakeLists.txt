add_lld_unittests(LLDCommonTests
  ErrorHandlerTest.cpp
)

target_link_libraries(LLDCommonTests
  PRIVATE
  lldCommon
)
