cmake_minimum_required(VERSION 3.12)

project(fdump-class-extents)

find_package(Clang REQUIRED)
# include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
# add_definitions(${LLVM_DEFINITIONS})

add_library(FDumpClassExtents SHARED fdump-class-extents.cpp)

set_target_properties(FDumpClassExtents PROPERTIES
    LINKER_LANGUAGE CXX
    COMPILE_FLAGS "-fno-rtti")

target_link_libraries(FDumpClassExtents PRIVATE
    clangAST
    clangASTMatchers
    clangBasic
    clangFrontend
    clangSerialization
    )

install(TARGETS FDumpClassExtents
    LIBRARY DESTINATION lib
    )
