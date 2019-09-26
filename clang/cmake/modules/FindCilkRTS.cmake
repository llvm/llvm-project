# 
#  CILKRTS_FOUND        -- cilk runtime was found.
#  CILKRTS_INCLUDE_DIR  -- directory with cilk runtime header file(s). 
#  CILKRTS_LIBRARY      -- full path to cilk runtime library. 
#  CILKRTS_LIBRARY_DIR  -- path to where the cilk runtime library is installed. 
#  CILKRTS_LINK_LIBS    -- set of link libraries (e.g. -lcilkrts) 
# 

message(STATUS "kitsune: looking for cilkrts...")

find_path(CILKRTS_INCLUDE_DIR cilk/cilk.h)
find_library(CILKRTS_LIBRARY cilkrts)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CilkRTS DEFAULT_MSG
				  CILKRTS_INCLUDE_DIR 
				  CILKRTS_LIBRARY)


if (CilkRTS_FOUND) 
  message(STATUS "kitsune: looking for cilkrts... FOUND")
  get_filename_component(CILKRTS_LIBRARY_DIR
                         ${CILKRTS_LIBRARY}
                         DIRECTORY
                         CACHE)
  set(KITSUNE_ENABLE_CILKRTS TRUE CACHE BOOL "Enable automatic include and library flags for Cilk RTS.")
  set(CILKRTS_LINK_LIBS "-lcilkrts" 
      CACHE STRING "List of libraries needed for cilkrts.")
  message(STATUS "        cilkrts include directory: ${CILKRTS_INCLUDE_DIR}")
  message(STATUS "        cilkrts library directory: ${CILKRTS_LIBRARY_DIR}")
  message(STATUS "        cilkrts link libraries   : ${CILKRTS_LINK_LIBS}")
else()
  message(STATUS "kitsune: looking for cilkrts... NOT FOUND")
  set(KITSUNE_ENABLE_CILKRTS FALSE CACHE BOOL "Enable automatic include and library flags for Cilk RTS.")
  set(CILKRTS_LINK_LIBS "" 
      CACHE STRING "List of libraries needed for cilkrts.")
endif()

#mark_as_advanced(CILKRTS_INCLUDE_DIR CILKRTS_LIBRARY CILKRTS_LIBRARY_DIR Cilkrts_FOUND)
