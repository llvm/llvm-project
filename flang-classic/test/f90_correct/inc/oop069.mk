#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop069  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop069.o:  $(SRC)/oop069.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -Hx,68,0x80 -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop069.f90 -o oop069.o

oop069: oop069.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop069.o fcheck.o $(LIBS) -o oop069

oop069.run: oop069
	@echo ------------------------------------ executing test oop069
	oop069
	-$(RM) shape_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
