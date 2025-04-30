#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop393  ########


oop393: run
	

build:  $(SRC)/oop393.f90
	-$(RM) oop393.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop393.f90 -o oop393.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop393.$(OBJX) check.$(OBJX) $(LIBS) -o oop393.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop393
	oop393.$(EXESUFFIX)

verify: ;

