#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

#  PGI default flags
#
#    FC_FLAGS := -fast
# 
#  Intel default flags
#
#    FC_FFLAGS := -O3 -xAVX -ftz  -ip  -no-fp-port -fp-model fast -no-prec-div 
#                  -no-prec -sqrt -override-limits -align array64byte 
#                  -DCPRINTEL -mkl
#
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


ALL_OBJS :=  kernel_binterp.o

all: build run verify

verify:  
	@(grep "FAIL" $(TEST).rslt && echo "FAILED") || (grep "PASSED" $(TEST).rslt -q && echo PASSED)

run: build
	@mkdir rundir ; if [ ! -d data ] ; then ln -s $(SRC)/data data &&  echo "symlinked data directory: ln -s $(SRC)/data data"; fi; cd rundir; ../kernel.exe >> ../$(TEST).rslt 2>&1 || ( echo RUN FAILED: DID NOT EXIT 0)
# symlink data/ so it can be found in the directory made by lit
	 @echo ----------------------run-ouput-was----------
	@cat $(TEST).rslt | grep -v "PASSED"

build: ${ALL_OBJS}
	${FC} ${FC_FLAGS}   -o kernel.exe $^

kernel_binterp.o: $(SRC_DIR)/kernel_binterp.F90
	${FC} ${FC_FLAGS} -c -o $@ $<

clean:
	rm -f *.exe *.mod *.o *.rslt
