#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test common_01  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

common_01.o:  $(SRC)/common_01.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) -O1 $(LDFLAGS) $(SRC)/common_01.f90 -o common_01.o

common_01: common_01.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) common_01.o fcheck.o $(LIBS) -o common_01

common_01.run: common_01
	@echo ------------------------------------ executing test common_01
	common_01
	-$(RM) shape_mod.mod test_shape_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ;

### End of Expected Targets ###
