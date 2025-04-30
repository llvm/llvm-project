#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop068  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop068.o:  $(SRC)/oop068.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop068.f90 -o oop068.o

oop068: oop068.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop068.o fcheck.o $(LIBS) -o oop068

oop068.run: oop068
	@echo ------------------------------------ executing test oop068
	oop068
	-$(RM) tmod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
