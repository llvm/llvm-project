#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop306  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop306.o:  $(SRC)/oop306.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop306.f90 -o oop306.o

oop306: oop306.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop306.o fcheck.o $(LIBS) -o oop306

oop306.run: oop306
	@echo ------------------------------------ executing test oop306
	oop306
	-$(RM) mod_gen.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
