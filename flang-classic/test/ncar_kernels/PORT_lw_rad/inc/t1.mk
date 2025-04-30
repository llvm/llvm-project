#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

#  PGI default flags
#
#    FC_FLAGS := -fast -Mipa=fast,inline
#
# 
#  Intel default flags
#
#	CPPDEFINES := -DOLD_SETCOEF -DOLD_RTRNMC -DOLD_CLDPRMC
#	#CPPDEFINES := -DOLD_RTRNMC -DOLD_CLDPRMC
#	#CPPDEFINES := -DOLD_RTRNMC
#	#CPPDEFINES := -DOLD_SETCOEF
#	#CPPDEFINES := 
#	#FC_FLAGS := ${CPPDEFINES} -xCORE-AVX2 -qopt-report=5 -no-opt-dynamic-align -O3 -fp-model fast=2 
#	FC_FLAGS := ${CPPDEFINES} -xHost -no-opt-dynamic-align -O3 -fp-model fast=2

#
FC_FLAGS := $(OPT)

ifeq ("$(FC)", "pgf90")
FC_FLAGS += -Mnofma
endif
ifeq ("$(FC)", "pgfortran")
FC_FLAGS += -Mnofma
endif
ifeq ("$(FC)", "flang")
endif
ifeq ("$(FC)", "gfortran")
endif
ifeq ("$(FC)", "ifort")
endif
ifeq ("$(FC)", "xlf")
endif


ALL_OBJS := kernel_driver.o radlw.o kgen_utils.o rrlw_kg08.o rrlw_kg15.o parrrtm.o rrlw_kg01.o rrlw_kg10.o rrlw_ref.o rrtmg_state.o rrlw_wvn.o rrtmg_lw_setcoef.o rrlw_kg16.o rrlw_kg02.o rrtmg_lw_cldprmc.o shr_kind_mod.o rrtmg_lw_rad.o rrtmg_lw_taumol.o rrlw_vsn.o rrlw_tbl.o rrlw_kg03.o ppgrid.o rrlw_kg07.o rrlw_kg14.o rrlw_kg04.o rrlw_kg12.o rrlw_kg13.o rrtmg_lw_rtrnmc.o rrlw_kg06.o rrlw_kg05.o rrlw_kg11.o rrlw_con.o rrlw_cld.o rrlw_kg09.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 radlw.o kgen_utils.o rrlw_kg08.o rrlw_kg15.o parrrtm.o rrlw_kg01.o rrlw_kg10.o rrlw_ref.o rrtmg_state.o rrlw_wvn.o rrtmg_lw_setcoef.o rrlw_kg16.o rrlw_kg02.o rrtmg_lw_cldprmc.o shr_kind_mod.o rrtmg_lw_rad.o rrtmg_lw_taumol.o rrlw_vsn.o rrlw_tbl.o rrlw_kg03.o ppgrid.o rrlw_kg07.o rrlw_kg14.o rrlw_kg04.o rrlw_kg12.o rrlw_kg13.o rrtmg_lw_rtrnmc.o rrlw_kg06.o rrlw_kg05.o rrlw_kg11.o rrlw_con.o rrlw_cld.o rrlw_kg09.o
	${FC} ${FC_FLAGS} -c -o $@ $<

radlw.o: $(SRC_DIR)/radlw.F90 kgen_utils.o rrtmg_lw_rad.o rrtmg_state.o shr_kind_mod.o ppgrid.o parrrtm.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg08.o: $(SRC_DIR)/rrlw_kg08.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg15.o: $(SRC_DIR)/rrlw_kg15.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

parrrtm.o: $(SRC_DIR)/parrrtm.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg01.o: $(SRC_DIR)/rrlw_kg01.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg10.o: $(SRC_DIR)/rrlw_kg10.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_ref.o: $(SRC_DIR)/rrlw_ref.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_state.o: $(SRC_DIR)/rrtmg_state.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_wvn.o: $(SRC_DIR)/rrlw_wvn.f90 kgen_utils.o parrrtm.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_setcoef.o: $(SRC_DIR)/rrtmg_lw_setcoef.F90 kgen_utils.o shr_kind_mod.o rrlw_vsn.o rrlw_wvn.o rrlw_ref.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg16.o: $(SRC_DIR)/rrlw_kg16.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg02.o: $(SRC_DIR)/rrlw_kg02.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_cldprmc.o: $(SRC_DIR)/rrtmg_lw_cldprmc.F90 kgen_utils.o shr_kind_mod.o parrrtm.o rrlw_vsn.o rrlw_cld.o rrlw_wvn.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_rad.o: $(SRC_DIR)/rrtmg_lw_rad.F90 kgen_utils.o shr_kind_mod.o parrrtm.o rrlw_con.o rrlw_wvn.o rrtmg_lw_cldprmc.o rrtmg_lw_setcoef.o rrtmg_lw_taumol.o rrtmg_lw_rtrnmc.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_taumol.o: $(SRC_DIR)/rrtmg_lw_taumol.f90 kgen_utils.o shr_kind_mod.o rrlw_vsn.o rrlw_wvn.o parrrtm.o rrlw_kg01.o rrlw_kg02.o rrlw_ref.o rrlw_con.o rrlw_kg03.o rrlw_kg04.o rrlw_kg05.o rrlw_kg06.o rrlw_kg07.o rrlw_kg08.o rrlw_kg09.o rrlw_kg10.o rrlw_kg11.o rrlw_kg12.o rrlw_kg13.o rrlw_kg14.o rrlw_kg15.o rrlw_kg16.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_vsn.o: $(SRC_DIR)/rrlw_vsn.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_tbl.o: $(SRC_DIR)/rrlw_tbl.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg03.o: $(SRC_DIR)/rrlw_kg03.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

ppgrid.o: $(SRC_DIR)/ppgrid.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg07.o: $(SRC_DIR)/rrlw_kg07.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg14.o: $(SRC_DIR)/rrlw_kg14.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg04.o: $(SRC_DIR)/rrlw_kg04.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg12.o: $(SRC_DIR)/rrlw_kg12.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg13.o: $(SRC_DIR)/rrlw_kg13.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_lw_rtrnmc.o: $(SRC_DIR)/rrtmg_lw_rtrnmc.F90 kgen_utils.o shr_kind_mod.o parrrtm.o rrlw_vsn.o rrlw_wvn.o rrlw_tbl.o rrlw_con.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg06.o: $(SRC_DIR)/rrlw_kg06.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg05.o: $(SRC_DIR)/rrlw_kg05.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg11.o: $(SRC_DIR)/rrlw_kg11.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_con.o: $(SRC_DIR)/rrlw_con.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_cld.o: $(SRC_DIR)/rrlw_cld.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrlw_kg09.o: $(SRC_DIR)/rrlw_kg09.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
