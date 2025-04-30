# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

########## Make rule for test oop719  ########


oop719: run
	

build:  $(SRC)/oop719.f90
	-$(RM) oop719.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop719.f90 -o oop719.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop719.$(OBJX) check.$(OBJX) $(LIBS) -o oop719.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop719
	oop719.$(EXESUFFIX)

verify: ;

