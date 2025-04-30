#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop426  ########


oop426: run
	

build:  $(SRC)/oop426.f90
	-$(RM) oop426.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop426.f90 -o oop426.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop426.$(OBJX) check.$(OBJX) $(LIBS) -o oop426.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop426
	oop426.$(EXESUFFIX)

verify: ;

