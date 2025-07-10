add_startup_object(
  tls
  SRC
    tls.cpp
)

add_startup_object(
  start
  SRC
    start.cpp
  DEPENDS
    libc.config.linux.app_h
)
