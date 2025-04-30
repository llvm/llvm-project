#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop314  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop314.o:  $(SRC)/oop314.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop314.f90 -o oop314.o

oop314: oop314.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop314.o fcheck.o $(LIBS) -o oop314

oop314.run: oop314
	@echo ------------------------------------ executing test oop314
	oop314
	-$(RM) mod_gen.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
