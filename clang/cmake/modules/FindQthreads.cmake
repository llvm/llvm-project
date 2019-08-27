# 
# 
#  QTHREADS_FOUND        -- qthreads was found.
#  QTHREADS_INCLUDE_DIR  -- directory with qthreads header file(s). 
#  QTHREADS_LIBRARY      -- full path to qthreads library. 
#  QTHREADS_LIBRARY_DIR  -- path to where the qthreads library is installed. 
#  QTHREADS_LINK_LIBS    -- set of link libraries (e.g. -lqthreads) 
# 

message(STATUS "kitsune: looking for qthreads...")

find_path(QTHREADS_INCLUDE_DIR qthread.h)
find_library(QTHREADS_LIBRARY qthread)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qthreads DEFAULT_MSG
				  QTHREADS_INCLUDE_DIR 
				  QTHREADS_LIBRARY)


if (Qthreads_FOUND) 
  message(STATUS "kitsune: looking for qthreads... FOUND")
  get_filename_component(QTHREADS_LIBRARY_DIR
                         ${QTHREADS_LIBRARY}
                         DIRECTORY
                         CACHE)
  set(KITSUNE_ENABLE_QTHREADS TRUE CACHE BOOL "Enable automatic include and library flags for Qthreads.")
  set(QTHREADS_LINK_LIBS "-lqthreads -lhwloc -lnuma -lpthread" 
      CACHE STRING "List of libraries needed for qthreads.")
else()
  message(STATUS "kitsune: looking for qthreads... NOT FOUND")
  set(KITSUNE_ENABLE_QTHREADS FALSE CACHE BOOL "Enable automatic include and library flags for Qthreads.")
  set(QTHREADS_LINK_LIBS "" 
      CACHE STRING "List of libraries needed for qthreads.")
endif()

#mark_as_advanced(QTHREADS_INCLUDE_DIR QTHREADS_LIBRARY QTHREADS_LIBRARY_DIR Qthreads_FOUND)
