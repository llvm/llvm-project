cmake_minimum_required(VERSION 3.14)

set (INCLUDEDIR include/)
set (SRCDIR src/)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -g -ggdb3")

project(CallGraph)

add_library(parser SHARED ${SRCDIR}/parser.cpp ${SRCDIR}/lib.cpp)
target_include_directories(parser PRIVATE ${INCLUDEDIR})

add_executable(CallGraph ${SRCDIR}/main.cpp)

target_include_directories(CallGraph PRIVATE ${INCLUDEDIR})
target_link_libraries(CallGraph PRIVATE parser)
