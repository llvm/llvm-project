#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop001  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop001.o:  $(SRC)/oop001.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop001.f90 -o oop001.o

oop001: clean oop001.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop001.o fcheck.o $(LIBS) -o oop001

oop001.run: oop001
	@echo ------------------------------------ executing test oop001
	oop001
	-$(RM) shape_mod.mod


### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
