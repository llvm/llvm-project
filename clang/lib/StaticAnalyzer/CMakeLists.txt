# These directories can significantly impact build time, only build
# them if anything depends on the clangStaticAnalyzer* libraries.
if(NOT CLANG_ENABLE_STATIC_ANALYZER)
  set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(Core)
add_subdirectory(Checkers)
add_subdirectory(Frontend)
