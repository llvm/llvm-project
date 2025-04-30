#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop394  ########


oop394: run
	

build:  $(SRC)/oop394.f90
	-$(RM) oop394.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop394.f90 -o oop394.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop394.$(OBJX) check.$(OBJX) $(LIBS) -o oop394.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop394
	oop394.$(EXESUFFIX)

verify: ;

