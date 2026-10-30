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
  dcgettext
  SRCS
    dcgettext.cpp
  HDRS
    dcgettext.h
)

add_entrypoint_object(
  dcgettext_l
  SRCS
    dcgettext_l.cpp
  HDRS
    dcgettext_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  dcngettext
  SRCS
    dcngettext.cpp
  HDRS
    dcngettext.h
)

add_entrypoint_object(
  dcngettext_l
  SRCS
    dcngettext_l.cpp
  HDRS
    dcngettext_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  dgettext
  SRCS
    dgettext.cpp
  HDRS
    dgettext.h
)

add_entrypoint_object(
  dgettext_l
  SRCS
    dgettext_l.cpp
  HDRS
    dgettext_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  dngettext
  SRCS
    dngettext.cpp
  HDRS
    dngettext.h
)

add_entrypoint_object(
  dngettext_l
  SRCS
    dngettext_l.cpp
  HDRS
    dngettext_l.h
  DEPENDS
    libc.hdr.types.locale_t
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
  ngettext
  SRCS
    ngettext.cpp
  HDRS
    ngettext.h
)

add_entrypoint_object(
  ngettext_l
  SRCS
    ngettext_l.cpp
  HDRS
    ngettext_l.h
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
