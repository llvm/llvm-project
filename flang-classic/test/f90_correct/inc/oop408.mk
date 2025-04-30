#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop408  ########


oop408: run
	

build:  $(SRC)/oop408.f90
	-$(RM) oop408.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop408.f90 -o oop408.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop408.$(OBJX) check.$(OBJX) $(LIBS) -o oop408.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop408
	oop408.$(EXESUFFIX)

verify: ;

