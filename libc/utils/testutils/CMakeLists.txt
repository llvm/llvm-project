if(CMAKE_HOST_UNIX)
  set(EFFile ExecuteFunctionUnix.cpp)
  set(FDReaderFile FDReaderUnix.cpp)
endif()

add_llvm_library(
  libc_test_utils
  RandUtils.cpp
  RandUtils.h
  StreamWrapper.cpp
  StreamWrapper.h
  ${EFFile}
  ExecuteFunction.h
  ${FDReaderFile}
  FDReader.h
)
