#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop461  ########


oop461: run
	

build:  $(SRC)/oop461.f90
	-$(RM) oop461.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop461.f90 -o oop461.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop461.$(OBJX) check.$(OBJX) $(LIBS) -o oop461.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop461
	oop461.$(EXESUFFIX)

verify: ;

