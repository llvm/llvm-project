cmake_minimum_required(VERSION 3.14)

set (INCLUDEDIR include/)
set (SRCDIR src/)

project(CallGraph)

add_executable(CallGraph ${SRCDIR}/main.cpp ${SRCDIR}/lib.cpp ${SRCDIR}/parser.cpp)

target_include_directories(CallGraph PRIVATE ${INCLUDEDIR})
