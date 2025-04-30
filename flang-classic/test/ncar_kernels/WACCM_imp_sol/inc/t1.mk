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

ALL_OBJS := kernel_driver.o mo_gas_phase_chemdr.o kgen_utils.o mo_tracname.o mo_nln_matrix.o mo_lu_solve.o chem_mods.o mo_prod_loss.o mo_lin_matrix.o ppgrid.o mo_imp_sol.o shr_kind_mod.o mo_lu_factor.o mo_indprd.o

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

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 mo_gas_phase_chemdr.o kgen_utils.o mo_tracname.o mo_nln_matrix.o mo_lu_solve.o chem_mods.o mo_prod_loss.o mo_lin_matrix.o ppgrid.o mo_imp_sol.o shr_kind_mod.o mo_lu_factor.o mo_indprd.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_gas_phase_chemdr.o: $(SRC_DIR)/mo_gas_phase_chemdr.F90 kgen_utils.o mo_imp_sol.o shr_kind_mod.o ppgrid.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_tracname.o: $(SRC_DIR)/mo_tracname.F90 kgen_utils.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_nln_matrix.o: $(SRC_DIR)/mo_nln_matrix.F90 kgen_utils.o shr_kind_mod.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lu_solve.o: $(SRC_DIR)/mo_lu_solve.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

chem_mods.o: $(SRC_DIR)/chem_mods.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_prod_loss.o: $(SRC_DIR)/mo_prod_loss.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lin_matrix.o: $(SRC_DIR)/mo_lin_matrix.F90 kgen_utils.o shr_kind_mod.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

ppgrid.o: $(SRC_DIR)/ppgrid.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_imp_sol.o: $(SRC_DIR)/mo_imp_sol.F90 kgen_utils.o ppgrid.o chem_mods.o shr_kind_mod.o mo_indprd.o mo_lin_matrix.o mo_nln_matrix.o mo_lu_factor.o mo_prod_loss.o mo_lu_solve.o mo_tracname.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lu_factor.o: $(SRC_DIR)/mo_lu_factor.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_indprd.o: $(SRC_DIR)/mo_indprd.F90 kgen_utils.o shr_kind_mod.o ppgrid.o chem_mods.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
