#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test oop144  ########

fcheck.o check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.o

oop144.o:  $(SRC)/oop144.f90 check_mod.mod
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop144.f90 -o oop144.o

oop144: oop144.o fcheck.o
	-$(FC) $(FFLAGS) $(LDFLAGS) oop144.o fcheck.o $(LIBS) -o oop144

oop144.run: oop144
	@echo ------------------------------------ executing test oop144
	oop144
	-$(RM) my_container.mod

### TA Expected Targets ###

build: $(TEST)

.PHONY: run
run: $(TEST).run

verify: ; 

### End of Expected Targets ###
