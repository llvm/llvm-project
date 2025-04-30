#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop391  ########


oop391: run
	

build:  $(SRC)/oop391.f90
	-$(RM) oop391.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop391.f90 -o oop391.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop391.$(OBJX) check.$(OBJX) $(LIBS) -o oop391.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop391
	oop391.$(EXESUFFIX)

verify: ;

