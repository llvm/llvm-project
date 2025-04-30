#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop421  ########


oop421: run
	

build:  $(SRC)/oop421.f90
	-$(RM) oop421.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop421.f90 -o oop421.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop421.$(OBJX) check.$(OBJX) $(LIBS) -o oop421.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop421
	oop421.$(EXESUFFIX)

verify: ;

