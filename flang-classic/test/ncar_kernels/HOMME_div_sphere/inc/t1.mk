#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

#
# PGI
#
#FC := pgf95
#FFLAGS := -O3
#
# Intel
#
# FC := pgfortran
# FFLAGS := -O3 -mmic -qopt-report=5 -fp-model fast
# FFLAGS := -O3 -xCORE-AVX2 -qopt-report=5 -fp-model fast
# FFLAGS := -O3 -xAVX -qopt-report=5 -fp-model fast
#
# GFORTRAN
# 
# FC :=gfortran
# FFLAGS := -O3 -ffree-form -ffree-line-length-none -D__GFORTRAN__ -I./
# #
#
# Cray 
#
#  FC := ftn
#  FFLAGS := -O2
#

FC_FLAGS := $(OPT)

ALL_OBJS := kernel_divergence_sphere.o

all: build run verify

verify:
	@echo "nothing to be done for verify"

run:
	mkdir rundir; cd rundir; ../kernel.exe

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_divergence_sphere.o: $(SRC_DIR)/kernel_divergence_sphere.F90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f *.exe *.optrpt *.o *.oo *.mod
