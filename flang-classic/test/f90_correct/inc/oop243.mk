#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop243  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop243.o:  $(SRC)/oop243.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop243.f90 -o oop243.o

oop243: oop243.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop243.o fcheck.o $(LIBS) -o oop243

oop243.run: oop243
	@echo ------------------------------------ executing test oop243
	oop243
	-$(RM) a.mod b.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
