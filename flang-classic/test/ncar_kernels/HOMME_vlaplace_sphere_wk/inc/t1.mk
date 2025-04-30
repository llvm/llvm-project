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
FC_FLAGS += -Mnofma

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


ALL_OBJS := kernel_driver.o viscosity_mod.o kgen_utils.o kinds.o shr_const_mod.o control_mod.o physical_constants.o parallel_mod.o shr_kind_mod.o element_mod.o gridgraph_mod.o derivative_mod.o coordinate_systems_mod.o physconst.o edge_mod.o dimensions_mod.o constituents.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 viscosity_mod.o kgen_utils.o kinds.o shr_const_mod.o control_mod.o physical_constants.o parallel_mod.o shr_kind_mod.o element_mod.o gridgraph_mod.o derivative_mod.o coordinate_systems_mod.o physconst.o edge_mod.o dimensions_mod.o constituents.o
	${FC} ${FC_FLAGS} -c -o $@ $<

viscosity_mod.o: $(SRC_DIR)/viscosity_mod.F90 kgen_utils.o derivative_mod.o element_mod.o kinds.o dimensions_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kinds.o: $(SRC_DIR)/kinds.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_const_mod.o: $(SRC_DIR)/shr_const_mod.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

control_mod.o: $(SRC_DIR)/control_mod.F90 kgen_utils.o kinds.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physical_constants.o: $(SRC_DIR)/physical_constants.F90 kgen_utils.o physconst.o
	${FC} ${FC_FLAGS} -c -o $@ $<

parallel_mod.o: $(SRC_DIR)/parallel_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

element_mod.o: $(SRC_DIR)/element_mod.F90 kgen_utils.o kinds.o coordinate_systems_mod.o dimensions_mod.o gridgraph_mod.o edge_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

gridgraph_mod.o: $(SRC_DIR)/gridgraph_mod.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

derivative_mod.o: $(SRC_DIR)/derivative_mod.F90 kgen_utils.o element_mod.o kinds.o dimensions_mod.o control_mod.o parallel_mod.o physical_constants.o
	${FC} ${FC_FLAGS} -c -o $@ $<

coordinate_systems_mod.o: $(SRC_DIR)/coordinate_systems_mod.F90 kgen_utils.o kinds.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physconst.o: $(SRC_DIR)/physconst.F90 kgen_utils.o shr_kind_mod.o shr_const_mod.o
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
