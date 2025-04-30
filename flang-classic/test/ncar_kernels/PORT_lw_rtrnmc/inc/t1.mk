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
#     -O2 -fp-model source -convert big_endian -assume byterecl -ftz
#     -traceback -assume realloc_lhs  -xAVX
#
#
FC_FLAGS := $(OPT)
FC_FLAGS += -Mnofma -Kieee

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


ALL_OBJS := kernel_driver.o rrtmg_lw_rad.o kgen_utils.o shr_kind_mod.o rrlw_vsn.o rrtmg_lw_rtrnmc.o parrrtm.o rrlw_wvn.o rrlw_tbl.o rrlw_con.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 rrtmg_lw_rad.o kgen_utils.o shr_kind_mod.o rrlw_vsn.o rrtmg_lw_rtrnmc.o parrrtm.o rrlw_wvn.o rrlw_tbl.o rrlw_con.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_rad.o: $(SRC_DIR)/rrtmg_lw_rad.f90 kgen_utils.o rrtmg_lw_rtrnmc.o shr_kind_mod.o parrrtm.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_vsn.o: $(SRC_DIR)/rrlw_vsn.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_rtrnmc.o: $(SRC_DIR)/rrtmg_lw_rtrnmc.f90 kgen_utils.o shr_kind_mod.o parrrtm.o rrlw_vsn.o rrlw_wvn.o rrlw_tbl.o rrlw_con.o
	${FC} ${FC_FLAGS} -c -o $@ $<

parrrtm.o: $(SRC_DIR)/parrrtm.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_wvn.o: $(SRC_DIR)/rrlw_wvn.f90 kgen_utils.o parrrtm.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_tbl.o: $(SRC_DIR)/rrlw_tbl.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_con.o: $(SRC_DIR)/rrlw_con.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.rslt
