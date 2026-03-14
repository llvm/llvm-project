add_compile_definitions(CMAKE_INSTALL_FULL_LIBDIR="${CMAKE_INSTALL_FULL_LIBDIR}")

add_subdirectory(Core)
add_subdirectory(Passes)
add_subdirectory(Profile)
add_subdirectory(Rewrite)
add_subdirectory(RuntimeLibs)
add_subdirectory(Target)
add_subdirectory(Utils)
