add_entrypoint_object(
  iswalpha
  SRCS
    iswalpha.cpp
  HDRS
    iswalpha.h
  DEPENDS
    libc.src.__support.wctype_utils    
)

add_entrypoint_object(
  iswdigit
  SRCS
    iswdigit.cpp
  HDRS
    iswdigit.h
  DEPENDS
    libc.src.__support.wctype_utils    
)


add_entrypoint_object(
  iswgraph
  SRCS
    iswgraph.cpp
  HDRS
    iswgraph.h
  DEPENDS
    libc.src.__support.wctype_utils    
)