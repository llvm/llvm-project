#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop088  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop088.o:  $(SRC)/oop088.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -Hx,68,0x80 -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop088.f90 -o oop088.o

oop088: oop088.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop088.o fcheck.o $(LIBS) -o oop088

oop088.run: oop088
	@echo ------------------------------------ executing test oop088
	oop088
	-$(RM) my_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
