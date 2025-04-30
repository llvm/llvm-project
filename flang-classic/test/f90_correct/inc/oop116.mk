#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop116  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop116.o:  $(SRC)/oop116.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop116.f90 -o oop116.o

oop116: oop116.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop116.o fcheck.o $(LIBS) -o oop116

oop116.run: oop116
	@echo ------------------------------------ executing test oop116
	oop116
	-$(RM) my_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
