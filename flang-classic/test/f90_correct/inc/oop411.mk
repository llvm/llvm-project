#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop411  ########


oop411: run
	

build:  $(SRC)/oop411.f90
	-$(RM) oop411.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop411.f90 -o oop411.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop411.$(OBJX) check.$(OBJX) $(LIBS) -o oop411.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop411
	oop411.$(EXESUFFIX)

verify: ;

