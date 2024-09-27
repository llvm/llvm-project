
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../FortranRuntime/include")

add_subdirectory(Common)
add_subdirectory(Support)
add_subdirectory(Evaluate)
add_subdirectory(Lower)
add_subdirectory(Parser)
add_subdirectory(Semantics)
add_subdirectory(Frontend)
add_subdirectory(FrontendTool)

add_subdirectory(Optimizer)
