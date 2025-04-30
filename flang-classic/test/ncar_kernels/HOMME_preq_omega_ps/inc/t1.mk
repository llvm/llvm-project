#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
#  PGI default flags
#
#    FC_FLAGS := -fast -Mipa=fast,inline
# 
#  Intel default flags
#
#  FC_FLAGS :=  -O2 -fp-model source -convert big_endian -assume byterecl
#               -ftz -traceback -assume realloc_lhs  -xAVX
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


ALL_OBJS := kernel_driver.o prim_advance_mod.o hybvcoord_mod.o prim_si_mod.o dimensions_mod.o kinds.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 prim_advance_mod.o hybvcoord_mod.o prim_si_mod.o dimensions_mod.o kinds.o
	${FC} ${FC_FLAGS} -c -o $@ $<

prim_advance_mod.o: $(SRC_DIR)/prim_advance_mod.F90 prim_si_mod.o kinds.o dimensions_mod.o hybvcoord_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

hybvcoord_mod.o: $(SRC_DIR)/hybvcoord_mod.F90 kinds.o dimensions_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

prim_si_mod.o: $(SRC_DIR)/prim_si_mod.F90 kinds.o dimensions_mod.o hybvcoord_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

dimensions_mod.o: $(SRC_DIR)/dimensions_mod.F90 
	${FC} ${FC_FLAGS} -c -o $@ $<

kinds.o: $(SRC_DIR)/kinds.F90 
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.rslt
