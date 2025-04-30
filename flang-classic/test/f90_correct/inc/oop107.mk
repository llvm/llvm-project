#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop107  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop107.o:  $(SRC)/oop107.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -Hx,68,0x80 -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop107.f90 -o oop107.o

oop107: oop107.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop107.o fcheck.o $(LIBS) -o oop107

oop107.run: oop107
	@echo ------------------------------------ executing test oop107
	oop107
	-$(RM) my_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
