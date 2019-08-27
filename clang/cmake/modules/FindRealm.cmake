# 
#  REALM_FOUND       -- kokkos was found.
#  REALM_INCLUDE_DIR -- directory with kokkos header files. 
#  REALM_LIBRARY     -- full path to the kokkos library. 
#  REALM_LIBRARY_DIR -- path to where the kokkos library is installed. 
#  REALM_LINK_LIBS   -- set of link libraries (e.g. -lrealm) 
# 

message(STATUS "kitsune: looking for realm...")

find_path(REALM_INCLUDE_DIR  realm.h)
find_library(REALM_LIBRARY realm)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Realm DEFAULT_MSG
				  REALM_INCLUDE_DIR 
				  REALM_LIBRARY)

if (Realm_FOUND) 
  message(STATUS "kitsune: looking for realm... FOUND")
  get_filename_component(REALM_LIBRARY_DIR
                         ${REALM_LIBRARY}
                         DIRECTORY
                         CACHE)
  set(KITSUNE_ENABLE_REALM TRUE CACHE BOOL "Enable automatic include and library flags for Realm.")
  set(REALM_LINK_LIBS "-lrealm" CACHE STRING "List of libraries need to link with for Realm.")
  message(STATUS "        realm include directory: ${REALM_INCLUDE_DIR}")
  message(STATUS "        realm library directory: ${REALM_LIBRARY_DIR}")
  message(STATUS "        realm link libraries   : ${REALM_LINK_LIBS}")
else()
  message(STATUS "kitsune: looking for realm... NOT FOUND")
  set(KITSUNE_ENABLE_REALM FALSE CACHE BOOL "Enable automatic include and library flags for Realm.")
  set(REALM_LINK_LIBS "" CACHE STRING "List of libraries need to link with for Realm.")
endif()

#mark_as_advanced(REALM_INCLUDE_DIR REALM_LIBRARY REALM_LIBRARY_DIR Realm_FOUND)
