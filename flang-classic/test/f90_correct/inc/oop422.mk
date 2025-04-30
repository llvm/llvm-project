#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop422  ########


oop422: run
	

build:  $(SRC)/oop422.f90
	-$(RM) oop422.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop422.f90 -o oop422.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop422.$(OBJX) check.$(OBJX) $(LIBS) -o oop422.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop422
	oop422.$(EXESUFFIX)

verify: ;

