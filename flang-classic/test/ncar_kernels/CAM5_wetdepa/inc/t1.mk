#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#  PGI default flags
#
#    FC_FLAGS := -fast -Mipa=fast,inline
# 
#  Intel default flags
#
#  FC_FLAGS := 
##   BASE = -mmic -vec-report=6 -fp-model fast -ftz -traceback
#   BASE = -qopt-report=5 -ftz -fp-model fast -traceback
# -02
#   FFLAGS = -O2 $(BASE)

# -O3
#   FFLAGS = -O3 $(BASE)

# -O3 -fast
#    FFLAGS = -O3 -fast -mmic $(BASE)
#
# Makefile for KGEN-generated kernel
FC_FLAGS := $(OPT)

ifeq ("$(FC)", "pgf90")
endif
ifeq ("$(FC)", "pgfortran")
endif
ifeq ("$(FC)", "flang")
endif
ifeq ("$(FC)", "gfortran")
endif
ifeq ("$(FC)", "ifort")
endif
ifeq ("$(FC)", "xlf")
endif


.SUFFIXES:
.SUFFIXES: .F90 .f90 .o
FPP := cpp
FPPFLAGS := -I. -traditional -P


OBJS  := wetdepa_driver.o wetdep.o kinds_mod.o params.o shr_const_mod.o shr_kind_mod.o
OBJS0 := wetdepa_driver.o wetdep_orig.o kinds_mod.o params.o shr_const_mod.o shr_kind_mod.o
ALL_OBJS :=$(OBJS)


run: build
	./kernel.exe

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

.F90.o:
	$(FC) $(FFLAGS) -c $<

#.F90.f90:
#	$(FPP) $(FPPFLAGS) $< >$*.f90 

wetdepa_driver.o: $(SRC_DIR)/wetdepa_driver.F90 shr_kind_mod.o wetdep.o
	${FC} ${FC_FLAGS} -c -o $@ $<

wetdep.o: $(SRC_DIR)/wetdep.F90 kinds_mod.o params.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.F90
	${FC} ${FC_FLAGS} -c -o $@ $<

kinds_mod.o: $(SRC_DIR)/kinds_mod.F90
	${FC} ${FC_FLAGS} -c -o $@ $<

params.o: $(SRC_DIR)/params.F90 shr_const_mod.o kinds_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_const_mod.o: $(SRC_DIR)/shr_const_mod.F90 shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

verify: run
	@echo "nothing to be done for verify"

clean:
	rm -rf *.o *.mod wetdepa_driver wetdepa_driver_v0 *.optrpt
