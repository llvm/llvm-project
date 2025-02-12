#.rst:
# FindTerminfo
# -----------
#
# Find a separate libtinfo (terminfo) library, if not included in ncurses.
# Some systems do not bundle them together, so we try to detect if it's needed.

if(TINFO_LIBRARIES)
  set(TERMINFO_FOUND TRUE)
else()
  find_library(TINFO_LIBRARIES NAMES tinfo DOC "The curses tinfo library" QUIET)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Terminfo
                                    FOUND_VAR
                                      TERMINFO_FOUND
                                    REQUIRED_VARS
                                      TINFO_LIBRARIES)
  if(TERMINFO_FOUND)
    mark_as_advanced(TERMINFO_FOUND)
  endif()
endif()
