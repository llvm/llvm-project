#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop403  ########


oop403: run
	

build:  $(SRC)/oop403.f90
	-$(RM) oop403.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop403.f90 -o oop403.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop403.$(OBJX) check.$(OBJX) $(LIBS) -o oop403.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop403
	oop403.$(EXESUFFIX)

verify: ;

