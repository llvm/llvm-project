#.rst:
# FindCursesAndPanel
# -----------
#
# Find the curses and panel library as a whole.

if(CURSES_INCLUDE_DIRS AND CURSES_LIBRARIES AND PANEL_LIBRARIES)
  set(CURSES_PANEL_FOUND TRUE)
else()
  find_package(Curses QUIET)
  find_library(PANEL_LIBRARIES NAMES panel DOC "The curses panel library" QUIET)
  if(CURSES_FOUND AND PANEL_LIBRARIES)
    mark_as_advanced(CURSES_INCLUDE_DIRS CURSES_LIBRARIES PANEL_LIBRARIES)
  endif()
endif()

