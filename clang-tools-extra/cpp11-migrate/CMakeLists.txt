set(LLVM_LINK_COMPONENTS support)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(tool)
add_subdirectory(Core)
