#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop464  ########


oop464: run
	

build:  $(SRC)/oop464.f90
	-$(RM) oop464.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop464.f90 -o oop464.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop464.$(OBJX) check.$(OBJX) $(LIBS) -o oop464.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop464
	oop464.$(EXESUFFIX)

verify: ;

