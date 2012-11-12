# TODO: Set the install directory.

set(known_subdirs
  "compiler-rt"
  "libcxx"
  )

foreach (dir ${known_subdirs})
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  endif()
endforeach()
