#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop311  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop311.o:  $(SRC)/oop311.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop311.f90 -o oop311.o

oop311: oop311.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop311.o fcheck.o $(LIBS) -o oop311

oop311.run: oop311
	@echo ------------------------------------ executing test oop311
	oop311
	-$(RM) mod_gen.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
