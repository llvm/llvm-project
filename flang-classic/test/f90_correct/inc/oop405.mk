#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop405  ########


oop405: run
	

build:  $(SRC)/oop405.f90
	-$(RM) oop405.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop405.f90 -o oop405.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop405.$(OBJX) check.$(OBJX) $(LIBS) -o oop405.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop405
	oop405.$(EXESUFFIX)

verify: ;

