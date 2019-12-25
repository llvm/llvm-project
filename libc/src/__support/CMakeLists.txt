add_gen_header(
  support_common_h
  DEF_FILE common.h.def
  PARAMS
    platform_defs=../../config/${LIBC_TARGET_OS}/platfrom_defs.h.inc
  GEN_HDR common.h
  DATA_FILES
    ../../config/${LIBC_TARGET_OS}/platfrom_defs.h.inc
)
