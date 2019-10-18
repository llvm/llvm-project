add_gen_header(
  support_common_h
  DEF_FILE common.h.def
  PARAMS
    entrypoint_macro=${LIBC_TARGET_OS}/entrypoint_macro.h.inc
  GEN_HDR common.h
  DATA_FILES
    ${LIBC_TARGET_OS}/entrypoint_macro.h.inc
)
