#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop049  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop049.o:  $(SRC)/oop049.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop049.f90 -o oop049.o

oop049: oop049.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop049.o fcheck.o $(LIBS) -o oop049

oop049.run: oop049
	@echo ------------------------------------ executing test oop049
	oop049
	-$(RM) shape_base_mod.mod shape_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
