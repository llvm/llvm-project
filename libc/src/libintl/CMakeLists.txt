add_entrypoint_object(
  bind_textdomain_codeset
  SRCS
    bind_textdomain_codeset.cpp
  HDRS
    bind_textdomain_codeset.h
)

add_entrypoint_object(
  bindtextdomain
  SRCS
    bindtextdomain.cpp
  HDRS
    bindtextdomain.h
)

add_entrypoint_object(
  gettext
  SRCS
    gettext.cpp
  HDRS
    gettext.h
)

add_entrypoint_object(
  gettext_l
  SRCS
    gettext_l.cpp
  HDRS
    gettext_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  textdomain
  SRCS
    textdomain.cpp
  HDRS
    textdomain.h
)
