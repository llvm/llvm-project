#.rst:
# FindCursesAndPanel
# -----------
#
# Find the curses, terminfo, and panel library as a whole.
# NOTE: terminfo and curses libraries are required separately, as
# some systems do not bundle them together.

if(CURSES_INCLUDE_DIRS AND CURSES_LIBRARIES AND TINFO_LIBRARIES AND PANEL_LIBRARIES)
  set(CURSESANDPANEL_FOUND TRUE)
else()
  find_package(Curses QUIET)
  find_package(TINFO_LIBRARIES NAMES tinfo DOC "The curses tinfo library" QUIET)
  find_library(PANEL_LIBRARIES NAMES panel DOC "The curses panel library" QUIET)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(CursesAndPanel
                                    FOUND_VAR
                                      CURSESANDPANEL_FOUND
                                    REQUIRED_VARS
                                      CURSES_INCLUDE_DIRS
                                      CURSES_LIBRARIES
                                      TINFO_LIBRARIES
                                      PANEL_LIBRARIES)
  if(CURSES_FOUND AND TINFO_LIBRARIES AND PANEL_LIBRARIES)
    mark_as_advanced(CURSES_INCLUDE_DIRS CURSES_LIBRARIES TINFO_LIBRARIES PANEL_LIBRARIES)
  endif()
endif()

