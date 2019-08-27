# 
# 
#  KOKKOS_FOUND       -- kokkos was found.
#  KOKKOS_INCLUDE_DIR -- directory with kokkos header files. 
#  KOKKOS_LIBRARY     -- full path to the kokkos library. 
#  KOKKOS_LIBRARY_DIR -- path to where the kokkos library is installed. 
#  KOKKOS_LINK_LIBS   -- set of link libraries (e.g. -lkokkos) 
# 

message(STATUS "kitsune: looking for kokkos...")

find_path(KOKKOS_INCLUDE_DIR Kokkos_Core.hpp)
find_library(KOKKOS_LIBRARY kokkos)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Kokkos DEFAULT_MSG
				  KOKKOS_INCLUDE_DIR 
				  KOKKOS_LIBRARY)

if (Kokkos_FOUND) 
  message(STATUS "kitsune: looking for kokkos... FOUND")
  get_filename_component(KOKKOS_LIBRARY_DIR
                         ${KOKKOS_LIBRARY}
                         DIRECTORY
                         CACHE)
  set(KITSUNE_ENABLE_KOKKOS TRUE CACHE BOOL "Enable automatic include and library flags for Kokkos.")
  set(KOKKOS_LINK_LIBS "-lkokkos -ldl -lrt" CACHE STRING "List of libraries need to link with for Kokkos.")
  message(STATUS "        kokkos include directory: ${KOKKOS_INCLUDE_DIR}")
  message(STATUS "        kokkos library directory: ${KOKKOS_LIBRARY_DIR}")
  message(STATUS "        kokkos link libraries   : ${KOKKOS_LINK_LIBS}")
else()
  message(STATUS "kitsune: looking for kokkos... NOT FOUND")
  set(KITSUNE_ENABLE_KOKKOS FALSE CACHE BOOL "Enable automatic include and library flags for Kokkos.")
  set(KOKKOS_LINK_LIBS "" CACHE STRING "List of libraries need to link with for Kokkos.")
endif()

#mark_as_advanced(KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY KOKKOS_LIBRARY_DIR Kokkos_FOUND)
