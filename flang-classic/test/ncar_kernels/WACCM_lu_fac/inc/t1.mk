#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Makefile for KGEN-generated kernel

#  PGI default flags
#
#    FC_FLAGS := -fast -Mipa=fast,inline
# 
#  Intel default flags
#
#    FC_FLAGS := -no-opt-dynamic-align  -convert big_endian -assume byterecl -ftz -traceback -assume realloc_lhs -fp-model source   -xHost  -O2
#

FC_FLAGS := $(OPT)

ALL_OBJS := kernel_driver.o mo_imp_sol.o kgen_utils.o chem_mods.o mo_lu_factor.o shr_kind_mod.o

all: build run verify

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 mo_imp_sol.o kgen_utils.o chem_mods.o mo_lu_factor.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_imp_sol.o: $(SRC_DIR)/mo_imp_sol.F90 kgen_utils.o mo_lu_factor.o shr_kind_mod.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

chem_mods.o: $(SRC_DIR)/chem_mods.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lu_factor.o: $(SRC_DIR)/mo_lu_factor.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
