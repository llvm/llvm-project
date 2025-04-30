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
#  FC_FLAGS :=  -fp-model source -convert big_endian -assume byterecl
#               -ftz -traceback -assume realloc_lhs -xHost -O2
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


ALL_OBJS := kernel_driver.o prim_advance_mod.o kgen_utils.o kinds.o shr_const_mod.o physical_constants.o shr_kind_mod.o prim_si_mod.o element_mod.o physconst.o coordinate_systems_mod.o gridgraph_mod.o edge_mod.o dimensions_mod.o constituents.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 prim_advance_mod.o kgen_utils.o kinds.o shr_const_mod.o physical_constants.o shr_kind_mod.o prim_si_mod.o element_mod.o physconst.o coordinate_systems_mod.o gridgraph_mod.o edge_mod.o dimensions_mod.o constituents.o
	${FC} ${FC_FLAGS} -c -o $@ $<

prim_advance_mod.o: $(SRC_DIR)/prim_advance_mod.F90 kgen_utils.o prim_si_mod.o kinds.o dimensions_mod.o element_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kinds.o: $(SRC_DIR)/kinds.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_const_mod.o: $(SRC_DIR)/shr_const_mod.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physical_constants.o: $(SRC_DIR)/physical_constants.F90 kgen_utils.o physconst.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

prim_si_mod.o: $(SRC_DIR)/prim_si_mod.F90 kgen_utils.o kinds.o dimensions_mod.o physical_constants.o
	${FC} ${FC_FLAGS} -c -o $@ $<

element_mod.o: $(SRC_DIR)/element_mod.F90 kgen_utils.o kinds.o coordinate_systems_mod.o dimensions_mod.o gridgraph_mod.o edge_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physconst.o: $(SRC_DIR)/physconst.F90 kgen_utils.o shr_kind_mod.o shr_const_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

coordinate_systems_mod.o: $(SRC_DIR)/coordinate_systems_mod.F90 kgen_utils.o kinds.o
	${FC} ${FC_FLAGS} -c -o $@ $<

gridgraph_mod.o: $(SRC_DIR)/gridgraph_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

edge_mod.o: $(SRC_DIR)/edge_mod.F90 kgen_utils.o kinds.o coordinate_systems_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

dimensions_mod.o: $(SRC_DIR)/dimensions_mod.F90 kgen_utils.o constituents.o
	${FC} ${FC_FLAGS} -c -o $@ $<

constituents.o: $(SRC_DIR)/constituents.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.rslt
