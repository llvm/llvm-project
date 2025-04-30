#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop151  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop151.o:  $(SRC)/oop151.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop151.f90 -o oop151.o

oop151: oop151.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop151.o fcheck.o $(LIBS) -o oop151

oop151.run: oop151
	@echo ------------------------------------ executing test oop151
	oop151
	-$(RM) my_container.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
