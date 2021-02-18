add_subdirectory(Common)
add_subdirectory(Evaluate)
add_subdirectory(Decimal)
add_subdirectory(Lower)
add_subdirectory(Parser)
add_subdirectory(Semantics)

if(FLANG_BUILD_NEW_DRIVER)
  add_subdirectory(Frontend)
  add_subdirectory(FrontendTool)
endif()

add_subdirectory(Optimizer)
