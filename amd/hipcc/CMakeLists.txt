cmake_minimum_required(VERSION 3.16.3)
project (hipcc)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set ( LINK_LIBS libstdc++fs.so)
add_executable(hipcc src/hipBin.cpp)
if (NOT WIN32)  #c++17 does not require the std lib linking
  target_link_libraries(hipcc ${LINK_LIBS} ) # for hipcc
endif()

project (hipconfig)
add_executable(hipconfig src/hipBin.cpp)
if (NOT WIN32)   #c++17 does not require the std lib linking
  target_link_libraries(hipconfig ${LINK_LIBS} ) # for hipconfig
endif()

set(HIP_VERSION_MAJOR 4 PARENT_SCOPE)
set(HIP_VERSION_MINOR 4 PARENT_SCOPE)
set(HIP_VERSION_PATCH 4 PARENT_SCOPE)
