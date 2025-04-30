#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
#  PGI default flags
#
#    FC := pgf95
#    FC_FLAGS := -O3
# 
#  Intel default flags
#
#    FC := pgfortran
#    FFLAGS := -O3 -xCORE-AVX2 -qopt-report=5 -fp-model fast
#    FFLAGS := -O3 -align array64byte  -xCORE-AVX2 -qopt-report=5 -fp-model fast=2
#    FFLAGS := -O3  -xCORE-AVX2 -qopt-report=5 -fp-model fast=2
#    FFLAGS := -O3 -align array64byte -xAVX -fp-model fast=2
#    FFLAGS := -O3 -align array64byte -mmic -qopt-report=5 -fp-model fast=2
#    FFLAGS := -O3 -xAVX -qopt-report=5 -fp-model fast=2
#
# GFORTRAN
# 
#    FC :=gfortran
#    FFLAGS := -O3 -ffree-form -ffree-line-length-none -D__GFORTRAN__ -I./
#
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



ALL_OBJS := kernel_gradient_sphere.o

verify: 
	@echo "nothing to be done for verify"

run: build
	./kernel.exe

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_gradient_sphere.o: $(SRC_DIR)/kernel_gradient_sphere.F90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.rslt
