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


# Makefile for KGEN-generated kernel

# Makefile for KGEN-generated kernel

ALL_OBJS := kernel_driver.o rrtmg_sw_rad.o kgen_utils.o rrtmg_sw_reftra.o rrsw_kg28.o rrsw_kg25.o rrsw_kg19.o parrrsw.o rrsw_tbl.o rrsw_kg21.o rrsw_kg23.o rrsw_con.o rrsw_wvn.o rrsw_kg27.o rrsw_kg24.o rrsw_kg16.o rrsw_vsn.o shr_kind_mod.o rrsw_kg17.o rrsw_kg20.o rrsw_kg29.o rrsw_kg22.o rrtmg_sw_taumol.o rrtmg_sw_vrtqdr.o rrsw_kg26.o rrsw_kg18.o rrtmg_sw_spcvmc.o

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 rrtmg_sw_rad.o kgen_utils.o rrtmg_sw_reftra.o rrsw_kg28.o rrsw_kg25.o rrsw_kg19.o parrrsw.o rrsw_tbl.o rrsw_kg21.o rrsw_kg23.o rrsw_con.o rrsw_wvn.o rrsw_kg27.o rrsw_kg24.o rrsw_kg16.o rrsw_vsn.o shr_kind_mod.o rrsw_kg17.o rrsw_kg20.o rrsw_kg29.o rrsw_kg22.o rrtmg_sw_taumol.o rrtmg_sw_vrtqdr.o rrsw_kg26.o rrsw_kg18.o rrtmg_sw_spcvmc.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_rad.o: $(SRC_DIR)/rrtmg_sw_rad.F90 kgen_utils.o rrtmg_sw_spcvmc.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_reftra.o: $(SRC_DIR)/rrtmg_sw_reftra.f90 kgen_utils.o shr_kind_mod.o rrsw_vsn.o rrsw_tbl.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg28.o: $(SRC_DIR)/rrsw_kg28.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg25.o: $(SRC_DIR)/rrsw_kg25.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg19.o: $(SRC_DIR)/rrsw_kg19.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

parrrsw.o: $(SRC_DIR)/parrrsw.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_tbl.o: $(SRC_DIR)/rrsw_tbl.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg21.o: $(SRC_DIR)/rrsw_kg21.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg23.o: $(SRC_DIR)/rrsw_kg23.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_con.o: $(SRC_DIR)/rrsw_con.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_wvn.o: $(SRC_DIR)/rrsw_wvn.f90 kgen_utils.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg27.o: $(SRC_DIR)/rrsw_kg27.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg24.o: $(SRC_DIR)/rrsw_kg24.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg16.o: $(SRC_DIR)/rrsw_kg16.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_vsn.o: $(SRC_DIR)/rrsw_vsn.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.o: $(SRC_DIR)/shr_kind_mod.f90 kgen_utils.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg17.o: $(SRC_DIR)/rrsw_kg17.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg20.o: $(SRC_DIR)/rrsw_kg20.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg29.o: $(SRC_DIR)/rrsw_kg29.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg22.o: $(SRC_DIR)/rrsw_kg22.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_taumol.o: $(SRC_DIR)/rrtmg_sw_taumol.f90 kgen_utils.o shr_kind_mod.o rrsw_vsn.o rrsw_kg16.o rrsw_con.o rrsw_wvn.o parrrsw.o rrsw_kg17.o rrsw_kg18.o rrsw_kg19.o rrsw_kg20.o rrsw_kg21.o rrsw_kg22.o rrsw_kg23.o rrsw_kg24.o rrsw_kg25.o rrsw_kg26.o rrsw_kg27.o rrsw_kg28.o rrsw_kg29.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_vrtqdr.o: $(SRC_DIR)/rrtmg_sw_vrtqdr.f90 kgen_utils.o shr_kind_mod.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg26.o: $(SRC_DIR)/rrsw_kg26.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrsw_kg18.o: $(SRC_DIR)/rrsw_kg18.f90 kgen_utils.o shr_kind_mod.o parrrsw.o
	${FC} ${FC_FLAGS} -c -o $@ $<

rrtmg_sw_spcvmc.o: $(SRC_DIR)/rrtmg_sw_spcvmc.f90 kgen_utils.o shr_kind_mod.o parrrsw.o rrtmg_sw_taumol.o rrsw_wvn.o rrsw_tbl.o rrtmg_sw_reftra.o rrtmg_sw_vrtqdr.o
	${FC} ${FC_FLAGS} -c -o $@ $<

kgen_utils.o: $(SRC_DIR)/kgen_utils.f90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
