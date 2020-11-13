# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

abt_sources += \
	arch/abtd_affinity.c \
	arch/abtd_affinity_parser.c \
	arch/abtd_env.c \
	arch/abtd_stream.c \
	arch/abtd_time.c \
	arch/abtd_ythread.c

if ABT_USE_FCONTEXT
abt_sources += \
	arch/fcontext/jump_@fctx_arch_bin@.S \
	arch/fcontext/make_@fctx_arch_bin@.S \
	arch/fcontext/take_@fctx_arch_bin@.S
endif
