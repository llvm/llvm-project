include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

# A target containing all code tweaks (i.e. mini-refactorings) provided by
# clangd.
# Built as an object library to make sure linker does not remove global
# constructors that register individual tweaks in a global registry.
# To enable these tweaks in exectubales or shared libraries, add
# $<TARGET_OBJECTS:obj.clangDaemonTweaks> to a list of sources, see
# clangd/tool/CMakeLists.txt for an example.
add_clang_library(clangDaemonTweaks OBJECT
  Dummy.cpp # FIXME: to avoid CMake errors due to empty inputs, remove when a
            #        first tweak lands.
  )
