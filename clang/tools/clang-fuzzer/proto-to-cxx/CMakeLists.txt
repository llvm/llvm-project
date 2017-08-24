set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})
set(CMAKE_CXX_FLAGS ${CXX_FLAGS_NOFUZZ})

# Hack to bypass LLVM's CMake source checks so we can have both a library and
# an executable built from this directory.
set(LLVM_OPTIONAL_SOURCES proto_to_cxx.cpp proto_to_cxx_main.cpp)

add_clang_library(clangProtoToCXX proto_to_cxx.cpp 
                  DEPENDS clangCXXProto
                  LINK_LIBS clangCXXProto
                  )

add_clang_executable(clang-proto-to-cxx proto_to_cxx_main.cpp)
target_link_libraries(clang-proto-to-cxx clangProtoToCXX)
