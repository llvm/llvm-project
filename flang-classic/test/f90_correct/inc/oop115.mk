#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop115  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop115.o:  $(SRC)/oop115.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop115.f90 -o oop115.o

oop115: oop115.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop115.o fcheck.o $(LIBS) -o oop115

oop115.run: oop115
	@echo ------------------------------------ executing test oop115
	oop115
	-$(RM) my_mod.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
