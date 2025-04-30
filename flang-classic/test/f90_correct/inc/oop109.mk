#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop109  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop109.o:  $(SRC)/oop109.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -Hx,68,0x80 -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop109.f90 -o oop109.o

oop109: oop109.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop109.o fcheck.o $(LIBS) -o oop109

oop109.run: oop109
	@echo ------------------------------------ executing test oop109
	oop109
	-$(RM) my_mod.mod my_mod2.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
