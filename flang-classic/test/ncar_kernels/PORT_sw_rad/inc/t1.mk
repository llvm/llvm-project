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
#    FC_FLAGS := -O1
#
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


# Makefile for KGEN-generated kernel

ALL_OBJS := kernel_driver.o radiation.o kgen_utils.o radsw.o rrsw_kg28.o rrtmg_state.o rrsw_kg25.o rrsw_kg19.o rrtmg_sw_reftra.o rrsw_cld.o parrrsw.o physics_types.o rrsw_tbl.o rrtmg_sw_rad.o rrsw_kg23.o cmparray_mod.o rrsw_con.o rrsw_wvn.o rrsw_kg27.o rrsw_ref.o rrsw_kg24.o rrsw_kg16.o rrsw_vsn.o scamMod.o constituents.o shr_const_mod.o shr_kind_mod.o rrtmg_sw_cldprmc.o rrsw_kg17.o radconstants.o rrsw_kg20.o rrsw_kg29.o rrsw_kg22.o mcica_subcol_gen_sw.o rrtmg_sw_taumol.o camsrfexch.o ppgrid.o rrtmg_sw_vrtqdr.o rrsw_kg26.o rrsw_kg18.o rrsw_kg21.o rrtmg_sw_spcvmc.o physconst.o mcica_random_numbers.o rrtmg_sw_setcoef.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 radiation.o kgen_utils.o radsw.o rrsw_kg28.o rrtmg_state.o rrsw_kg25.o rrsw_kg19.o rrtmg_sw_reftra.o rrsw_cld.o parrrsw.o physics_types.o rrsw_tbl.o rrtmg_sw_rad.o rrsw_kg23.o cmparray_mod.o rrsw_con.o rrsw_wvn.o rrsw_kg27.o rrsw_ref.o rrsw_kg24.o rrsw_kg16.o rrsw_vsn.o scamMod.o constituents.o shr_const_mod.o shr_kind_mod.o rrtmg_sw_cldprmc.o rrsw_kg17.o radconstants.o rrsw_kg20.o rrsw_kg29.o rrsw_kg22.o mcica_subcol_gen_sw.o rrtmg_sw_taumol.o camsrfexch.o ppgrid.o rrtmg_sw_vrtqdr.o rrsw_kg26.o rrsw_kg18.o rrsw_kg21.o rrtmg_sw_spcvmc.o physconst.o mcica_random_numbers.o rrtmg_sw_setcoef.o
	${FC} ${FC_FLAGS} -c -o $@ $<

radiation.o: $(SRC_DIR)/radiation.F90 kgen_utils.o radsw.o ppgrid.o shr_kind_mod.o parrrsw.o rrtmg_state.o physics_types.o camsrfexch.o radconstants.o
	${FC} ${FC_FLAGS} -c -o $@ $<

radsw.o: $(SRC_DIR)/radsw.F90 kgen_utils.o shr_kind_mod.o ppgrid.o parrrsw.o rrtmg_state.o scamMod.o cmparray_mod.o mcica_subcol_gen_sw.o rrtmg_sw_rad.o physconst.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg28.o: $(SRC_DIR)/rrsw_kg28.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_state.o: $(SRC_DIR)/rrtmg_state.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg25.o: $(SRC_DIR)/rrsw_kg25.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg19.o: $(SRC_DIR)/rrsw_kg19.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_reftra.o: $(SRC_DIR)/rrtmg_sw_reftra.f90 kgen_utils.o shr_kind_mod.o rrsw_vsn.o rrsw_tbl.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_cld.o: $(SRC_DIR)/rrsw_cld.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

parrrsw.o: $(SRC_DIR)/parrrsw.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physics_types.o: $(SRC_DIR)/physics_types.F90 kgen_utils.o ppgrid.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_tbl.o: $(SRC_DIR)/rrsw_tbl.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_rad.o: $(SRC_DIR)/rrtmg_sw_rad.F90 kgen_utils.o shr_kind_mod.o parrrsw.o rrsw_con.o rrtmg_sw_cldprmc.o rrtmg_sw_setcoef.o rrtmg_sw_spcvmc.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg23.o: $(SRC_DIR)/rrsw_kg23.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

cmparray_mod.o: $(SRC_DIR)/cmparray_mod.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_con.o: $(SRC_DIR)/rrsw_con.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_wvn.o: $(SRC_DIR)/rrsw_wvn.f90 kgen_utils.o parrrsw.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg27.o: $(SRC_DIR)/rrsw_kg27.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_ref.o: $(SRC_DIR)/rrsw_ref.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg24.o: $(SRC_DIR)/rrsw_kg24.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg16.o: $(SRC_DIR)/rrsw_kg16.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_vsn.o: $(SRC_DIR)/rrsw_vsn.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

scamMod.o: $(SRC_DIR)/scamMod.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

constituents.o: $(SRC_DIR)/constituents.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_const_mod.o: $(SRC_DIR)/shr_const_mod.F90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_cldprmc.o: $(SRC_DIR)/rrtmg_sw_cldprmc.F90 kgen_utils.o shr_kind_mod.o parrrsw.o rrsw_vsn.o rrsw_wvn.o rrsw_cld.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg17.o: $(SRC_DIR)/rrsw_kg17.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

radconstants.o: $(SRC_DIR)/radconstants.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg20.o: $(SRC_DIR)/rrsw_kg20.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg29.o: $(SRC_DIR)/rrsw_kg29.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg22.o: $(SRC_DIR)/rrsw_kg22.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mcica_subcol_gen_sw.o: $(SRC_DIR)/mcica_subcol_gen_sw.f90 kgen_utils.o shr_kind_mod.o parrrsw.o mcica_random_numbers.o rrsw_wvn.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_taumol.o: $(SRC_DIR)/rrtmg_sw_taumol.f90 kgen_utils.o shr_kind_mod.o rrsw_vsn.o rrsw_kg16.o rrsw_con.o rrsw_wvn.o parrrsw.o rrsw_kg17.o rrsw_kg18.o rrsw_kg19.o rrsw_kg20.o rrsw_kg21.o rrsw_kg22.o rrsw_kg23.o rrsw_kg24.o rrsw_kg25.o rrsw_kg26.o rrsw_kg27.o rrsw_kg28.o rrsw_kg29.o
	${FC} ${FC_FLAGS} -c -o $@ $<

camsrfexch.o: $(SRC_DIR)/camsrfexch.F90 kgen_utils.o shr_kind_mod.o ppgrid.o constituents.o
	${FC} ${FC_FLAGS} -c -o $@ $<

ppgrid.o: $(SRC_DIR)/ppgrid.F90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_vrtqdr.o: $(SRC_DIR)/rrtmg_sw_vrtqdr.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg26.o: $(SRC_DIR)/rrsw_kg26.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg18.o: $(SRC_DIR)/rrsw_kg18.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg21.o: $(SRC_DIR)/rrsw_kg21.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_spcvmc.o: $(SRC_DIR)/rrtmg_sw_spcvmc.f90 kgen_utils.o shr_kind_mod.o parrrsw.o rrtmg_sw_taumol.o rrsw_wvn.o rrsw_tbl.o rrtmg_sw_reftra.o rrtmg_sw_vrtqdr.o
	${FC} ${FC_FLAGS} -c -o $@ $<

physconst.o: $(SRC_DIR)/physconst.F90 kgen_utils.o shr_kind_mod.o shr_const_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mcica_random_numbers.o: $(SRC_DIR)/mcica_random_numbers.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_setcoef.o: $(SRC_DIR)/rrtmg_sw_setcoef.f90 kgen_utils.o shr_kind_mod.o rrsw_ref.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
