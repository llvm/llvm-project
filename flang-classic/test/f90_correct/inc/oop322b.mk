#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop322b  ########


oop322b: run
	

build:  $(SRC)/oop322b.f90
	-$(RM) oop322b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop322b.f90 -o oop322b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop322b.$(OBJX) check.$(OBJX) $(LIBS) -o oop322b.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop322b
	oop322b.$(EXESUFFIX)

verify: ;

