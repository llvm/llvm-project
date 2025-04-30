#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop774  ########


oop774: run
	

build:  $(SRC)/oop774.f90
	-$(RM) oop774.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop774.f90 -o oop774.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop774.$(OBJX) check.$(OBJX) $(LIBS) -o oop774.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop774
	oop774.$(EXESUFFIX)

verify: ;

