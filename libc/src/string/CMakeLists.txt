add_entrypoint_object(
  strcat
  SRCS
    strcat.cpp
  HDRS
    strcat.h
  DEPENDS
    strcpy
    string_h
)

add_entrypoint_object(
  strcpy
  SRCS
    strcpy.cpp
  HDRS
    strcpy.h
  DEPENDS
    string_h
)

add_subdirectory(memory_utils)
