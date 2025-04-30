#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop262  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop262.o:  $(SRC)/oop262.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop262.f90 -o oop262.o

oop262: oop262.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop262.o fcheck.o $(LIBS) -o oop262

oop262.run: oop262
	@echo ------------------------------------ executing test oop262
	oop262
	-$(RM) mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
