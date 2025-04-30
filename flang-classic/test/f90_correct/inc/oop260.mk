#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop260  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop260.o:  $(SRC)/oop260.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop260.f90 -o oop260.o

oop260: oop260.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop260.o fcheck.o $(LIBS) -o oop260

oop260.run: oop260
	@echo ------------------------------------ executing test oop260
	oop260
	-$(RM) mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
