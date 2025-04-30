#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop401  ########


oop401: run
	

build:  $(SRC)/oop401.f90
	-$(RM) oop401.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop401.f90 -o oop401.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop401.$(OBJX) check.$(OBJX) $(LIBS) -o oop401.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop401
	oop401.$(EXESUFFIX)

verify: ;

