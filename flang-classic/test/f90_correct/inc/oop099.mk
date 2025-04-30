#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop099  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop099.o:  $(SRC)/oop099.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop099.f90 -o oop099.o

oop099: oop099.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop099.o fcheck.o $(LIBS) -o oop099

oop099.run: oop099
	@echo ------------------------------------ executing test oop099
	oop099
	-$(RM) z.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
