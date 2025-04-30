#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Makefile for KGEN-generated kernel
#
#
#  PGI default flags
# FC_FLAGS := -fast -Mipa=fast,inline
#
#
#  Intel default flags
# FC_FLAGS := -O3 -xHost


FC_FLAGS := $(OPT)

ALL_OBJS := kernel_driver.o mo_psrad_interface.o mo_lrtm_kgs.o mo_cld_sampling.o mo_lrtm_solver.o mo_rrtm_coeffs.o mo_exception_stub.o mo_physical_constants.o mo_radiation_parameters.o mo_kind.o mo_spec_sampling.o mo_random_numbers.o mo_lrtm_setup.o mo_math_constants.o mo_rrtm_params.o mo_rad_fastmath.o mo_lrtm_driver.o mo_lrtm_gas_optics.o

all: build run verify

verify: 
	@(grep "verification.FAIL" $(TEST).rslt && echo "FAILED") || (grep "verification.PASS" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	 @cat $(TEST).rslt

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS} -o kernel.exe $^

kernel_driver.o: $(SRC_DIR)/kernel_driver.f90 mo_psrad_interface.o mo_lrtm_kgs.o mo_cld_sampling.o mo_lrtm_solver.o mo_rrtm_coeffs.o mo_exception_stub.o mo_physical_constants.o mo_radiation_parameters.o mo_kind.o mo_spec_sampling.o mo_random_numbers.o mo_lrtm_setup.o mo_math_constants.o mo_rrtm_params.o mo_rad_fastmath.o mo_lrtm_driver.o mo_lrtm_gas_optics.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_psrad_interface.o: $(SRC_DIR)/mo_psrad_interface.f90 mo_lrtm_driver.o mo_rrtm_params.o mo_kind.o mo_spec_sampling.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lrtm_kgs.o: $(SRC_DIR)/mo_lrtm_kgs.f90 mo_kind.o mo_rrtm_params.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_cld_sampling.o: $(SRC_DIR)/mo_cld_sampling.f90 mo_kind.o mo_random_numbers.o mo_exception_stub.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lrtm_solver.o: $(SRC_DIR)/mo_lrtm_solver.f90 mo_kind.o mo_rrtm_params.o mo_rad_fastmath.o mo_math_constants.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_rrtm_coeffs.o: $(SRC_DIR)/mo_rrtm_coeffs.f90 mo_kind.o mo_rrtm_params.o mo_lrtm_kgs.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_exception_stub.o: $(SRC_DIR)/mo_exception_stub.f90 
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_physical_constants.o: $(SRC_DIR)/mo_physical_constants.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_radiation_parameters.o: $(SRC_DIR)/mo_radiation_parameters.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_kind.o: $(SRC_DIR)/mo_kind.f90 
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_spec_sampling.o: $(SRC_DIR)/mo_spec_sampling.f90 mo_kind.o mo_random_numbers.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_random_numbers.o: $(SRC_DIR)/mo_random_numbers.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lrtm_setup.o: $(SRC_DIR)/mo_lrtm_setup.f90 mo_rrtm_params.o mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_math_constants.o: $(SRC_DIR)/mo_math_constants.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_rrtm_params.o: $(SRC_DIR)/mo_rrtm_params.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_rad_fastmath.o: $(SRC_DIR)/mo_rad_fastmath.f90 mo_kind.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lrtm_driver.o: $(SRC_DIR)/mo_lrtm_driver.f90 mo_rrtm_params.o mo_kind.o mo_spec_sampling.o mo_radiation_parameters.o mo_lrtm_setup.o mo_cld_sampling.o mo_rrtm_coeffs.o mo_lrtm_gas_optics.o mo_lrtm_kgs.o mo_physical_constants.o mo_lrtm_solver.o
	${FC} ${FC_FLAGS} -c -o $@ $<

mo_lrtm_gas_optics.o: $(SRC_DIR)/mo_lrtm_gas_optics.f90 mo_kind.o mo_lrtm_setup.o mo_lrtm_kgs.o mo_exception_stub.o
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f kernel.exe *.mod *.o *.oo *.rslt
