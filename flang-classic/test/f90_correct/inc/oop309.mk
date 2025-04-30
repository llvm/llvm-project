#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop309  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop309.o:  $(SRC)/oop309.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop309.f90 -o oop309.o

oop309: oop309.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop309.o fcheck.o $(LIBS) -o oop309

oop309.run: oop309
	@echo ------------------------------------ executing test oop309
	oop309
	-$(RM) mod_gen.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
