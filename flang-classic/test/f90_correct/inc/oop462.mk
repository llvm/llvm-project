#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop462  ########


oop462: run
	

build:  $(SRC)/oop462.f90
	-$(RM) oop462.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop462.f90 -o oop462.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop462.$(OBJX) check.$(OBJX) $(LIBS) -o oop462.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop462
	oop462.$(EXESUFFIX)

verify: ;

