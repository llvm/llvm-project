# Copyright (C) 2013-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

# Makefile for benchmark tests.  The only useful target here is `bench`.
# Add benchmark functions in alphabetical order.

subdir := benchtests

include ../Makeconfig
bench-math := acos acosh asin asinh atan atanh cos cosh exp exp2 log log2   \
	      modf pow rint sin sincos sinh sqrt tan tanh fmin fmax fminf   \
	      fmaxf powf trunc truncf roundeven roundevenf expf exp2f logf  \
	      log2f sincosf sinf cosf isnan isinf isfinite hypot logb logbf \
	      exp10f ilogb ilogbf cbrt erf erfc exp10 expm1 j0 j1 lgamma    \
	      log10 log1p tgamma y0 y1 atan2

ifneq (,$(filter yes,$(float96-fcts)))
bench-math += cbrtl
endif

ifneq (,$(filter yes,$(float128-fcts) $(float128-alias-fcts)))
bench-math += expf128 powf128 sinf128 ilogbf128
endif

bench-pthread := pthread_once thread_create pthread-locks

bench-string := ffs ffsll

ifeq (${BENCHSET},)
bench := $(bench-math) $(bench-pthread) $(bench-string)
else
bench := $(foreach B,$(filter bench-%,${BENCHSET}), ${${B}})
endif

# String function benchmarks.
string-benchset := memccpy memchr memcmp memcpy memmem memmove \
		   mempcpy memset rawmemchr stpcpy stpncpy strcasecmp strcasestr \
		   strcat strchr strchrnul strcmp strcpy strcspn strlen \
		   strncasecmp strncat strncmp strncpy strnlen strpbrk strrchr \
		   strspn strstr strcpy_chk stpcpy_chk memrchr strsep strtok \
		   strcoll memcpy-large memcpy-random memmove-large memset-large \
		   memcpy-walk memset-walk memmove-walk

# Build and run locale-dependent benchmarks only if we're building natively.
ifeq (no,$(cross-compiling))
wcsmbs-benchset := wcslen wcsnlen wcscpy wcpcpy wcsncpy wcpncpy wcscat wcsncat \
		   wcscmp wcsncmp wcschr wcschrnul wcsrchr wcsspn wcspbrk wcscspn \
		   wmemchr wmemset wmemcmp
else
wcsmbs-benchset :=
endif

string-benchset-all := $(string-benchset) ${wcsmbs-benchset}

ifeq (no,$(cross-compiling))
# We have to generate locales
LOCALES := en_US.UTF-8 tr_TR.UTF-8 cs_CZ.UTF-8 fa_IR.UTF-8 fr_FR.UTF-8 \
	   ja_JP.UTF-8 si_LK.UTF-8 en_GB.UTF-8 vi_VN.UTF-8 ar_SA.UTF-8 \
	   da_DK.UTF-8 pl_PL.UTF-8 pt_PT.UTF-8 el_GR.UTF-8 ru_RU.UTF-8 \
	   he_IL.UTF-8 is_IS.UTF-8 es_ES.UTF-8 hi_IN.UTF-8 sv_SE.UTF-8 \
	   hu_HU.UTF-8 it_IT.UTF-8 sr_RS.UTF-8 zh_CN.UTF-8
include ../gen-locales.mk
endif

stdlib-benchset := strtod

stdio-common-benchset := sprintf

math-benchset := math-inlines

ifeq (${BENCHSET},)
benchset := $(string-benchset-all) $(stdlib-benchset) $(stdio-common-benchset) \
	    $(math-benchset)
else
benchset := $(foreach B,$(filter %-benchset,${BENCHSET}), ${${B}})
endif

CFLAGS-bench-ffs.c += -fno-builtin
CFLAGS-bench-ffsll.c += -fno-builtin
CFLAGS-bench-sqrt.c += -fno-builtin
CFLAGS-bench-fmin.c += -fno-builtin
CFLAGS-bench-fminf.c += -fno-builtin
CFLAGS-bench-fmax.c += -fno-builtin
CFLAGS-bench-fmaxf.c += -fno-builtin
CFLAGS-bench-trunc.c += -fno-builtin
CFLAGS-bench-truncf.c += -fno-builtin
CFLAGS-bench-roundeven.c += -fno-builtin
CFLAGS-bench-roundevenf.c += -fno-builtin
CFLAGS-bench-isnan.c += -fsignaling-nans
CFLAGS-bench-isinf.c += -fsignaling-nans
CFLAGS-bench-isfinite.c += -fsignaling-nans

ifeq (${BENCHSET},)
bench-malloc := malloc-thread malloc-simple
else
bench-malloc := $(filter malloc-%,${BENCHSET})
endif

$(addprefix $(objpfx)bench-,$(bench-math)): $(libm)
$(addprefix $(objpfx)bench-,$(math-benchset)): $(libm)
$(addprefix $(objpfx)bench-,$(bench-pthread)): $(shared-thread-library)
$(addprefix $(objpfx)bench-,$(bench-malloc)): $(shared-thread-library)
$(addprefix $(objpfx)bench-,pthread-locks): $(libm)



# Rules to build and execute the benchmarks.  Do not put any benchmark
# parameters beyond this point.

# We don't want the benchmark programs to run in parallel since that could
# affect their performance.
.NOTPARALLEL:

bench-extra-objs = json-lib.o

extra-objs += $(bench-extra-objs)
others-extras = $(bench-extra-objs)

include ../Rules

binaries-bench := $(addprefix $(objpfx)bench-,$(bench))
binaries-benchset := $(addprefix $(objpfx)bench-,$(benchset))
binaries-bench-malloc := $(addprefix $(objpfx)bench-,$(bench-malloc))

# The default duration: 1 seconds.
ifndef BENCH_DURATION
BENCH_DURATION := 1
endif

CPPFLAGS-nonlib += -DDURATION=$(BENCH_DURATION) -D_ISOMAC

# Use clock_gettime to measure performance of functions.  The default is
# to use the architecture-specific high precision timing instructions.
ifdef USE_CLOCK_GETTIME
CPPFLAGS-nonlib += -DUSE_CLOCK_GETTIME
else
# On x86 processors, use RDTSCP, instead of RDTSC, to measure performance
# of functions.  All x86 processors since 2010 support RDTSCP instruction.
ifdef USE_RDTSCP
CPPFLAGS-nonlib += -DUSE_RDTSCP
endif
endif

DETAILED_OPT :=

ifdef DETAILED
DETAILED_OPT := -d
endif

bench-deps := bench-skeleton.c bench-timing.h Makefile

run-bench = $(test-wrapper-env) \
	    $(run-program-env) \
	    $($*-ENV) $(test-via-rtld-prefix) $${run}

timing-type := $(objpfx)bench-timing-type

# This makes sure CPPFLAGS-nonlib and CFLAGS-nonlib are passed
# for all these modules.
cpp-srcs-left := $(binaries-benchset:=.c) $(binaries-bench:=.c) \
		 $(binaries-bench-malloc:=.c) $(timing-type:=.c)
lib := nonlib
include $(patsubst %,$(..)libof-iterator.mk,$(cpp-srcs-left))

bench-clean:
	rm -f $(binaries-bench) $(addsuffix .o,$(binaries-bench))
	rm -f $(binaries-benchset) $(addsuffix .o,$(binaries-benchset))
	rm -f $(binaries-bench-malloc) $(addsuffix .o,$(binaries-bench-malloc))
	rm -f $(timing-type) $(addsuffix .o,$(timing-type))
	rm -f $(addprefix $(objpfx),$(bench-extra-objs))

# Validate the passed in BENCHSET
ifneq ($(strip ${BENCHSET}),)
VALIDBENCHSETNAMES := bench-pthread bench-math bench-string string-benchset \
   wcsmbs-benchset stdlib-benchset stdio-common-benchset math-benchset \
   malloc-thread malloc-simple
INVALIDBENCHSETNAMES := $(filter-out ${VALIDBENCHSETNAMES},${BENCHSET})
ifneq (${INVALIDBENCHSETNAMES},)
$(info The following values in BENCHSET are invalid: ${INVALIDBENCHSETNAMES})
$(info The valid ones are: ${VALIDBENCHSETNAMES})
$(error Invalid BENCHSET value)
endif
endif

bench: bench-build bench-set bench-func bench-malloc

# Target to only build the benchmark without running it.  We generate locales
# only if we're building natively.
ifeq (no,$(cross-compiling))
bench-build: $(gen-locales) $(timing-type) $(binaries-bench) \
	$(binaries-benchset) $(binaries-bench-malloc)
else
bench-build: $(timing-type) $(binaries-bench) $(binaries-benchset) \
	$(binaries-bench-malloc)
endif

bench-set: $(binaries-benchset)
	for run in $^; do \
	  echo "Running $${run}"; \
	  $(run-bench) > $${run}.out; \
	done

bench-malloc: $(binaries-bench-malloc)
	for run in $^; do \
	  echo "$${run}"; \
	  if [ `basename $${run}` = "bench-malloc-thread" ]; then \
		for thr in 1 8 16 32; do \
			echo "Running $${run} $${thr}"; \
			$(run-bench) $${thr} > $${run}-$${thr}.out; \
		done;\
	  else \
		for thr in 8 16 32 64 128 256 512 1024 2048 4096; do \
		  echo "Running $${run} $${thr}"; \
		  $(run-bench) $${thr} > $${run}-$${thr}.out; \
		done;\
	  fi;\
	done

# Build and execute the benchmark functions.  This target generates JSON
# formatted bench.out.  Each of the programs produce independent JSON output,
# so one could even execute them individually and process it using any JSON
# capable language or tool.
bench-func: $(binaries-bench)
	if [ -n '$^' ] ; then \
	{ timing_type=$$($(test-wrapper-env) \
			 $(run-program-env) \
			 $(test-via-rtld-prefix) \
			 $(timing-type)); \
	  echo "{\"timing_type\": \"$${timing_type}\","; \
	  echo " \"functions\": {"; \
	  for run in $^; do \
	    if ! [ "x$${run}" = "x$<" ]; then \
	      echo ","; \
	    fi; \
	    echo "Running $${run}" >&2; \
	    $(run-bench) $(DETAILED_OPT); \
	  done; \
	  echo; \
	  echo " }"; \
	  echo "}"; \
	  } > $(objpfx)bench.out-tmp; \
	  if [ -f $(objpfx)bench.out ]; then \
	    mv -f $(objpfx)bench.out $(objpfx)bench.out.old; \
	  fi; \
	  mv -f $(objpfx)bench.out-tmp $(objpfx)bench.out; \
	  $(PYTHON) scripts/validate_benchout.py $(objpfx)bench.out \
	  scripts/benchout.schema.json; \
	fi

ifeq ($(bind-now),yes)
link-bench-bind-now = -Wl,-z,now
endif

bench-link-targets = $(timing-type) $(binaries-bench) $(binaries-benchset) \
	$(binaries-bench-malloc)

$(bench-link-targets): %: %.o $(objpfx)json-lib.o \
	$(link-extra-libs-tests) \
  $(sort $(filter $(common-objpfx)lib%,$(link-libc))) \
  $(addprefix $(csu-objpfx),start.o) $(+preinit) $(+postinit)
	$(+link-tests)

$(bench-link-targets): LDFLAGS += $(link-bench-bind-now)

$(objpfx)bench-%.c: %-inputs $(bench-deps)
	{ if [ -n "$($*-INCLUDE)" ]; then \
	  cat $($*-INCLUDE); \
	fi; \
	$(PYTHON) scripts/bench.py $(patsubst %-inputs,%,$<); } > $@-tmp
	mv -f $@-tmp $@
