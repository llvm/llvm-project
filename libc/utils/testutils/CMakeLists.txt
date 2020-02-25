add_library(
  libc_test_utils
  ExecuteFunction.h
)

if(CMAKE_HOST_UNIX)
  target_sources(libc_test_utils PRIVATE ExecuteFunctionUnix.cpp)
endif()
